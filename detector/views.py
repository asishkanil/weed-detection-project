
import base64
import logging
from io import BytesIO
from pathlib import Path

from django.shortcuts import render, redirect
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "training_results"
    / "weed_model_v2"
    / "weights"
    / "best.pt"
)

model = None


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        model = YOLO(str(MODEL_PATH))

    return model


# Page 1: Image upload
def home(request):
    return render(request, "detector/home.html")


# Process the uploaded image
def detect(request):
    if request.method != "POST":
        return redirect("home")

    image_file = request.FILES.get("image")

    if not image_file:
        request.session["detection_error"] = (
            "Please select an image first."
        )
        return redirect("home")

    if image_file.size > 10 * 1024 * 1024:
        request.session["detection_error"] = (
            "Image must be smaller than 10 MB."
        )
        return redirect("home")

    try:
        confidence = float(
            request.POST.get("confidence", "25")
        ) / 100

        if not 0.10 <= confidence <= 0.90:
            request.session["detection_error"] = (
                "Choose a confidence threshold between 10% and 90%."
            )
            return redirect("home")

        image_file.seek(0)

        with Image.open(image_file) as img:
            img.verify()

        image_file.seek(0)

        with Image.open(image_file) as img:
            if img.width * img.height > 20_000_000:
                request.session["detection_error"] = (
                    "Image dimensions are too large."
                )
                return redirect("home")

            image = img.convert("RGB")

        yolo_model = get_model()

        results = yolo_model.predict(
            source=image,
            device="cpu",
            conf=confidence,
            verbose=False,
        )

        result = results[0]

        # Draw the predicted bounding boxes.
        plotted_image = result.plot()
        annotated_image = Image.fromarray(
            plotted_image[:, :, ::-1]
        )

        buffer = BytesIO()
        annotated_image.save(buffer, format="JPEG", quality=92)

        image_data = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())

                detections.append({
                    "class_name": result.names[class_id],
                    "confidence": round(score * 100, 2),
                })

        weed_count = sum(
            1 for item in detections
            if item["class_name"].lower() == "weed"
        )

        wheat_count = sum(
            1 for item in detections
            if item["class_name"].lower() == "wheat"
        )

        # Store results in the server-side Django session.
        request.session["detection_result"] = {
            "image": image_data,
            "detections": detections,
            "count": len(detections),
            "weed_count": weed_count,
            "wheat_count": wheat_count,
            "confidence_threshold": round(confidence * 100),
        }

        request.session.pop("detection_error", None)

        return redirect("results")

    except (UnidentifiedImageError, OSError, ValueError):
        request.session["detection_error"] = (
            "Please upload a valid image file."
        )
        return redirect("home")

    except Exception:
        logging.exception("Weed detection failed")
        request.session["detection_error"] = (
            "Detection failed. Check the VS Code terminal."
        )
        return redirect("home")


# Page 2: Display the detection output
def results(request):
    detection_result = request.session.get("detection_result")

    if not detection_result:
        return redirect("home")

    return render(
        request,
        "detector/results.html",
        {"result": detection_result},
    )


# Page 3: Weed knowledge and management
def weed_management(request):
    return render(request, "detector/weed_management.html")
