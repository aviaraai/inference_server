# Godhaar Inference Pipeline

This repository contains the pure Machine Learning pipeline for the Godhaar cattle re-identification system. 

It is designed to be a completely standalone AI service. It handles all the heavy mathematical lifting and image processing, while leaving the business rules (like GPS distance checks and final Match/Review decisions) to the main backend API.

Here is exactly how our pipeline works.

## The AI Pipeline

When an API request comes in (either to register a new cow or search for an existing one), the images flow through the following steps:

### 1. The Quality Gate
Before any heavy AI runs, we check the raw images for blur and exposure. If a field officer takes a bad, blurry photo, the pipeline instantly rejects it. This prevents garbage data from polluting our database.

### 2. YOLO Detection and Cropping
We use YOLOv8 to scan the image and find the cow. The pipeline ensures there is exactly one cow in the frame, and then strictly crops the image around the muzzle.

### 3. DINOv2 Embedding
The cropped muzzle image is passed into our fine-tuned GodhaarModel (based on DINOv2). The model converts the image of the cow into a 256-dimensional mathematical vector (embedding).

### 4. FAISS Vector Database
We use FAISS to store and search these embeddings. 
- During a Registration, the 256-dimensional vector is saved into FAISS, and its ID is safely recorded in a local SQLite database.
- During a Search, FAISS compares the new vector against thousands of stored vectors in milliseconds to find the closest matches.

### 5. Color Extraction
While the muzzle is being embedded, the pipeline also looks at the front/body images provided. It uses our rule-based color classifiers to extract the body coat color (like Black, Brown, Spotted) and the muzzle skin color.

## Integration Notes

This pipeline intentionally does not make the final decision. It simply returns the top matching candidates, their mathematical similarity scores, and their extracted colors.

The main backend API takes these results, calculates the GPS distance, applies any color penalties, and uses the final thresholds to decide if it is a MATCH or if it needs MANUAL REVIEW.

## Running the Pipeline

You can run the entire pipeline locally using Docker. The Docker container expects a few external files to be mounted:
- The DINOv2 model checkpoint file.
- The YOLO weights file.
- The directory where the FAISS index and SQLite database will be saved.
- The Wildlife color rule scripts.
