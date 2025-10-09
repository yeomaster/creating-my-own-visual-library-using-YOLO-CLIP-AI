![img_019](https://github.com/user-attachments/assets/992da4d0-cd54-4e13-b596-66efd251da49)![img_006](https://github.com/user-attachments/assets/404eb67d-2ba1-4714-b325-ded0cd95ddb7)# creating-my-own-visual-library-using-YOLO-CLIP-AI

**This project was made as an attempt to create my own visual library.**

I had used tools such as **YOLO**, **CLIP** and **speech_recognition** in an attempt to not only make a visual library that worked in tandem to YOLO, but also activate it via voice command

**NOTE**: the voice command is still currently under development

How each library was used:

**1. YOLO:**
YOLO v8 nano was the most simplest to implement, as it is a pretrained model that already draws bounding boxes on objects it is already trained to recognise, that being a total of 80 items.
However, I found that 80 items to be quite limiting and required many more items to be identified for future projects. As such, I had wanted to a create a visual library of my own.
Therefore, YOLO acted as my base/main recognition model, helping identifying objects it knows with relatively good accuracy, while adding my own visual library on top


**2. CLIP AI**
This is where it gets complicated.
CLIP is a pretrained model that maps an image (or crop) to a numeric vector (“embedding”).
We save those vectors for our enrolled examples and later compare a new image’s vector to them using cosine similarity (dot product after L2-normalization). The higher the cosine score (closer to +1), the more similar the images.
2-1: Enrollment: when I manually add ~20 pictures for a label, CLIP converts each picture into a 512-D vector. These vectors are stacked into a matrix (and stored alongside their labels).
2-2: Recognition: during live video, each candidate region (a cropped patch of the frame) is passed through CLIP to get a vector. That vector is compared (by our code) to all saved vectors using cosine similarity.
If the best similarity exceeds a threshold, we treat it as that label and report the score.
(Note: CLIP does not draw bounding boxes; boxes/regions come from the rest of the pipeline. CLIP’s role is just the vector/embedding.)
CLIP is essential in not just creating a library but also assisting in identifying live feed items to items in the visual lbrary


**3. IndexFace**
This library is used to recognise, draw bounding boxes and embed/convert to vectors the largest face it sees (ie the users face)
This is many used for facial/ user recognition
No real purpose for this library, just though it'll be fun to try


**4. indexing**
After you finish taking pictures, you run index. This step turns every saved picture into a short numeric fingerprint (for items we use CLIP; for faces we use InsightFace) and stores all those fingerprints together with their labels. Later, during live video, the app takes a small crop of what it sees, makes the same kind of fingerprint for that crop, and then compares it to every fingerprint in your saved set. The result is a similarity score (from about –1 to +1, but you’ll usually see 0 to 0.9). The closer the score is to +1, the more it looks like something you saved; if the best score is high enough, the app shows that saved label, and if not, it treats it as unknown


**How to run this code:**
NOTE: I had created this code via **Visual Studio COde** as such you need to run the following commands in order inside the terminal

1. python visual_library_app.py enroll --mode face --label (name of person)
2. python visual_library_app.py enroll --mode item --label (name of item)
3.python visual_library_app.py enroll --mode item --label (item YOLO cant recognise) --manual   
4. python visual_library_app.py index
5. python visual_library_app.py recognize --face-thresh 0.35 --item-thresh 0.7


The following is some of the pictures I used in my visual library that I added manually because YOLO did not recognise them:

Chair:

![img_000](https://github.com/user-attachments/assets/f2d44a42-3275-4b0f-af5f-f8a09be61dec)
![img_001](https://github.com/user-attachments/assets/aed04ea6-e026-4def-bbdd-35f34fe4d778)

Book:

![img_004](https://github.com/user-attachments/assets/e2a2d5cc-ca72-40f0-adcf-abf147a79b53)
![img_015](https://github.com/user-attachments/assets/6298299c-67fb-46e3-bd06-58515635c111)

Pokeball:

![img_006](https://github.com/user-attachments/assets/f8eb6bd5-7ac0-4d6f-8dc6-502489d0b663)
![img_019](https://github.com/user-attachments/assets/fd5687f7-e7f6-454f-89e2-506e344f176d)

And here is the result:


Note: I will not be uploading any videos or pictures of my face for obvious reasons
