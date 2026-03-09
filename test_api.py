import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_create_task():
    url = f"{BASE_URL}/ml-tasks/trigger/"
    import os
    file_path = os.path.abspath("test_polyp.jpg")
    payload = {
        "task_type": "polyp_detect",
        "file_id": file_path
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code in [201, 202]:
            print("✅ Task created successfully:")
            print(json.dumps(response.json(), indent=2))
            return response.json().get('task_id')
        else:
            print(f"❌ Failed to create task. Status: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def test_explain_task(task_id):
    if not task_id:
        print("Skipping explain test (no task_id)")
        return
        
    url = f"{BASE_URL}/ml-tasks/{task_id}/explain/"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            print("✅ Explanation generated successfully:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to generate explanation. Status: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Connection error: {e}")

def test_list_tasks():
    url = f"{BASE_URL}/ml-tasks/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Task list retrieved successfully:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to list tasks. Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    print("Testing API...")
    task_id = test_create_task()
    if task_id:
        print("\nTesting Explanation...")
        test_explain_task(task_id)
    print("\n")
    test_list_tasks()
