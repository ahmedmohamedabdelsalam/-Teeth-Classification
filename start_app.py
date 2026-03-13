import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure the script runs from the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add common python paths
    sys.path.append(script_dir)
    
    print("="*60)
    print("DENTAL IMAGING SYSTEM - LAUNCHER")
    print("="*60)
    print("Starting backend server...")
    from app import config
    print(f"Application will be available at: http://localhost:{config.PORT}")
    print("="*60)
    
    try:
        from app.main import app
        uvicorn.run(app, host=config.HOST, port=config.PORT)
    except ImportError as e:
        print(f"Error: Missing dependencies? {e}")
        print("Please run: pip install -r requirements.txt")
    except Exception as e:
        print(f"An error occurred: {e}")
