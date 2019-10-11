from flask import Flask, request
from modelProcess import detectObject
import cv2
app = Flask(__name__)

@app.route("/")
def main():
    return "app is working"


@app.route("/processImg", methods=["POST"])
def processReq():
    data = request.files["img"]
    data.save("img.jpg")
    img = cv2.imread("img.jpg")
    detectObject(img)
    return "er"

if __name__ == "__main__":
    app.run(debug=True)