import gradio as gr
from fastai.vision.all import load_learner, PILImage

learn = load_learner('models/export.pkl')

labels = learn.dls.vocab


def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


gr.Interface(
    fn=predict,
    inputs=gr.components.Image(shape=(512, 512)),
    outputs=gr.components.Label(num_top_classes=12)
).launch(share=True)
