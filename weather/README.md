{
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "/Users/myhome/Desktop",
          "/Users/myhome/Downloads"
        ]
      }
    }
  }



# step2 

{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/myhome/Desktop",
        "/Users/myhome/Downloads"
      ]
    },
    "weather": {
      "command": "/Users/myhome/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/myhome/weather",
        "run",
        "weather.py"
      ]
    }
  }
}


# steps 👍

pip install uv
which uv

uv init
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx
python weather.py


# step 5 : 
replace you API  : 


Go to https://openweathermap.org/
Click "Sign Up" or "Sign In" if you already have an account
After signing in, go to your account page
Navigate to the "API Keys" tab
You can either use the default key that's generated or create a new one
Copy the API key and replace YOUR_API_KEY_HERE in the code with your actual ke