import os

def create_project_structure():
    """
    Initializes the directory hierarchy for the Road Damage Detection project.
    Ensures that all output and data paths exist before execution.
    """
    
    # List of required directories
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'deployment',
        'models',
        'outputs/csv',
        'outputs/weights',
        'outputs/plots',
        'outputs/samples'
    ]

    print("🚀 Initializing project workspace...")

    for folder in directories:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created/Verified: {folder}")
        except Exception as e:
            print(f"❌ Error creating {folder}: {e}")

    print("\n✨ Workspace ready. You can now place your datasets in 'data/raw'.")

if __name__ == "__main__":
    create_project_structure()
