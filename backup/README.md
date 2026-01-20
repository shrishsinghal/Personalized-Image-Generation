# stable-diffusion-webapp

#Vanilla Installation using Gradio and ngrok for local deployment and testing on your system

# Clone the repository
git clone https://github.com/<your-username>/stable-diffusion-webapp.git

# Install dependencies
cd backend
pip install -r requirements.txt

# Run Flask backend
python app.py

# Launch Gradio frontend
cd frontend
python web_app.py
