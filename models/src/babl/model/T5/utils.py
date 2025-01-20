



def clean(x):
    return x.replace("<pad>", "").replace("</s>", "").strip().lower()

