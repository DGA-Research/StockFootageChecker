from google.cloud import vision

client = vision.ImageAnnotatorClient()
# any small jpg/png you have locally
with open("sample.jpg", "rb") as f:
    img = vision.Image(content=f.read())

resp = client.web_detection(image=img)
wd = resp.web_detection

print("Web entities:", [(e.description, getattr(e, "score", None)) for e in getattr(wd, "web_entities", [])][:5])
print("Similar images:", len(getattr(wd, "visually_similar_images", [])))
print("Pages with matches:", len(getattr(wd, "pages_with_matching_images", [])))
