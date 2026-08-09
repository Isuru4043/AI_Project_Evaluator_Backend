import os
import requests
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("SONAR_HOST_URL", "https://sonarcloud.io")
organization = os.getenv("SONAR_ORG_KEY")
token = os.getenv("SONAR_TOKEN")

if not organization or not token:
    print("SONAR_ORG_KEY or SONAR_TOKEN is missing in the environment.")
    exit(1)

session = requests.Session()
session.auth = (token, "")

def list_projects():
    response = session.get(
        f"{host}/api/projects/search",
        params={"organization": organization, "ps": 500},
        timeout=30
    )
    response.raise_for_status()
    return response.json().get("components", [])

def delete_project(project_key):
    response = session.post(
        f"{host}/api/projects/delete",
        data={"project": project_key},
        timeout=30
    )
    response.raise_for_status()

def main():
    print(f"Fetching projects for organization: {organization}...")
    projects = list_projects()
    
    if not projects:
        print("No projects found to delete.")
        return
        
    print(f"Found {len(projects)} projects. Starting deletion...")
    for p in projects:
        project_key = p['key']
        print(f"Deleting project: {project_key}...")
        try:
            delete_project(project_key)
            print(f"  -> Successfully deleted {project_key}")
        except Exception as e:
            print(f"  -> Failed to delete {project_key}: {e}")
            
    print("All done!")

if __name__ == "__main__":
    main()
