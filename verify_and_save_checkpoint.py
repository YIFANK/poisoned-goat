"""
Script to verify and properly save checkpoints from Google Colab.
Run this in Colab after training completes to ensure checkpoints are saved correctly.
"""

import os
import shutil
import zipfile
from pathlib import Path


def verify_checkpoint(checkpoint_dir):
    """Verify that a checkpoint directory contains valid files with correct sizes."""
    print(f"Verifying checkpoint: {checkpoint_dir}")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Required files
    required_files = {
        "adapter_config.json": (1 * 1024, 10 * 1024),  # 1 KB to 10 KB
        "adapter_model.safetensors": (50 * 1024 * 1024, 200 * 1024 * 1024),  # 50 MB to 200 MB
    }
    
    # Also check for .bin format
    adapter_bin = "adapter_model.bin"
    if os.path.exists(os.path.join(checkpoint_dir, adapter_bin)):
        required_files[adapter_bin] = (50 * 1024 * 1024, 200 * 1024 * 1024)  # 50 MB to 200 MB
    
    all_valid = True
    
    for filename, (min_size, max_size) in required_files.items():
        file_path = os.path.join(checkpoint_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ {filename}: NOT FOUND")
            all_valid = False
            continue
        
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size < min_size:
            print(f"❌ {filename}: {file_size_mb:.2f} MB ({file_size:,} bytes)")
            print(f"   ⚠️  TOO SMALL! Expected at least {min_size / (1024*1024):.2f} MB")
            all_valid = False
        elif file_size > max_size:
            print(f"⚠️  {filename}: {file_size_mb:.2f} MB ({file_size:,} bytes)")
            print(f"   Larger than expected (max {max_size / (1024*1024):.2f} MB), but might be OK")
        else:
            print(f"✅ {filename}: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    # Check for optional files
    optional_files = ["README.md", "tokenizer_config.json", "special_tokens_map.json"]
    print("\nOptional files:")
    for filename in optional_files:
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✅ {filename}: {file_size / 1024:.2f} KB")
    
    print("=" * 60)
    
    if all_valid:
        print("✅ Checkpoint is valid!")
    else:
        print("❌ Checkpoint has issues! Files are missing or too small.")
        print("   This usually means:")
        print("   1. Training didn't complete successfully")
        print("   2. Model wasn't saved correctly")
        print("   3. Files were corrupted during save")
    
    return all_valid


def save_to_drive(checkpoint_dir, drive_path="/content/drive/MyDrive/checkpoints"):
    """Save checkpoint to Google Drive (most reliable method)."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except ImportError:
        print("❌ Not running in Google Colab. Cannot use Drive.")
        return False
    except Exception as e:
        print(f"❌ Error mounting Drive: {e}")
        return False
    
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
    drive_checkpoint = os.path.join(drive_path, checkpoint_name)
    
    print(f"\nSaving checkpoint to Google Drive...")
    print(f"Source: {checkpoint_dir}")
    print(f"Destination: {drive_checkpoint}")
    
    try:
        # Create directory in Drive
        os.makedirs(drive_path, exist_ok=True)
        
        # Remove existing directory if it exists
        if os.path.exists(drive_checkpoint):
            print(f"Removing existing directory: {drive_checkpoint}")
            shutil.rmtree(drive_checkpoint)
        
        # Copy checkpoint to Drive
        shutil.copytree(checkpoint_dir, drive_checkpoint)
        
        # Verify copy
        safetensors_path = os.path.join(drive_checkpoint, "adapter_model.safetensors")
        if os.path.exists(safetensors_path):
            drive_size = os.path.getsize(safetensors_path) / (1024 * 1024)
            print(f"✅ Saved to Drive: {drive_size:.2f} MB")
            print(f"✅ Location: {drive_checkpoint}")
            return True
        else:
            print("❌ File not found in Drive after copy!")
            return False
            
    except Exception as e:
        print(f"❌ Error saving to Drive: {e}")
        return False


def create_zip(checkpoint_dir, zip_path=None):
    """Create a zip file of the checkpoint for easier download."""
    if zip_path is None:
        checkpoint_name = os.path.basename(checkpoint_dir.rstrip('/'))
        zip_path = f"{checkpoint_name}.zip"
    
    print(f"\nCreating zip file: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, checkpoint_dir)
                    zipf.write(file_path, arcname)
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if file_size_mb > 1:  # Only print files larger than 1 MB
                        print(f"  Added: {file} ({file_size_mb:.2f} MB)")
        
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"✅ Zip file created: {zip_size_mb:.2f} MB")
        print(f"✅ Location: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"❌ Error creating zip: {e}")
        return None


def upload_to_huggingface(checkpoint_dir, repo_id, token=None):
    """Upload checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, upload_folder
        
        print(f"\nUploading to HuggingFace Hub: {repo_id}")
        
        api = HfApi()
        upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        
        print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_and_save_checkpoint.py <checkpoint_dir> [--drive] [--zip] [--hf <repo_id>]")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    
    # Verify checkpoint
    is_valid = verify_checkpoint(checkpoint_dir)
    
    if not is_valid:
        print("\n⚠️  Checkpoint is not valid! Please check the training completed successfully.")
        sys.exit(1)
    
    # Save to Drive if requested
    if "--drive" in sys.argv:
        save_to_drive(checkpoint_dir)
    
    # Create zip if requested
    if "--zip" in sys.argv:
        create_zip(checkpoint_dir)
    
    # Upload to HuggingFace if requested
    if "--hf" in sys.argv:
        hf_idx = sys.argv.index("--hf")
        if hf_idx + 1 < len(sys.argv):
            repo_id = sys.argv[hf_idx + 1]
            upload_to_huggingface(checkpoint_dir, repo_id)
        else:
            print("❌ --hf requires a repo_id argument")

