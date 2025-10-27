Hello guys,
what i did
1. Made a zero shot calssification(script.py) [model used=facebook/bart-large-mnli]
2. I got labelled data. (too large to upload, sent it in the group)
3. I took only those rows where the confidence level is above 88%.(filter.py)
4. Extrinsic and intrinsic so much disbalanced so my model was more bent towards extrinsic.
5. I used data augmentation. final file --> (aug.csv)
6. I used aug.csv data to fine tune (model: distilbert-base-uncased)
7. app.py is optinal. i created a streamlit app for simplicity.
