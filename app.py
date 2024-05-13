from transformers import pipeline
import cv2
import math
import gradio as gr
from PIL import Image
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def chop_image(img_path, vertical_thresh):
    img = cv2.imread(img_path)
    # convert to black and white
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    vertical_threshold = int(img.shape[0] * vertical_thresh)
    final_lines = []
    skip_lines = []
    if lines is None:
        return [img]
    for i in range(len(lines)):
        rho1 = lines[i][0][0]
        close_lines = []
        for j in range(i+1, len(lines)):
            rho2 = lines[j][0][0]
            if i not in skip_lines and j not in skip_lines:
                if abs(rho2 - rho1) < vertical_threshold:
                    close_lines.append(lines[i][0])
                    skip_lines.append(j)
                if len(close_lines) == 0:
                    final_lines.append(lines[i][0])
                else:
                    best_line = [math.inf, math.inf]
                    for close_line in close_lines:
                        if close_line[0] < best_line[0]:
                            best_line = close_line
                    final_lines.append(best_line)
    final_final_lines = np.unique(np.array(final_lines), axis=0)
    chopped_images = []
    for line in range(len(final_final_lines)):
        if line == 0:
            y0 = 0
        else:
            y0 = int(final_final_lines[line - 1][0])
        y1 = int(final_final_lines[line][0])
        # print(y0, y1)
        # img = cv2.imread(im_path)

        # plt.imsave(f"line{line}.png", img[y0:y1, :, :])
        chopped_images.append(img[y0:y1, :, :])
    return chopped_images

def image_to_text(image_list):
    single_line_text_images = []
    output_text = ""
    for image in image_list:
        print(image)
        single_line_text_images += chop_image(image, 0.1)
    print(len(single_line_text_images))
    for img in single_line_text_images:
        pil_im = Image.fromarray(img)
        text = pipe(pil_im)[0]["generated_text"]
        output_text += f"{text}\n\n"
        print()
    return output_text

def text_processing(texts, similarity_thresh, min_words, max_words):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(min_words, max_words)) # setup TFIDF
    X = vectorizer.fit_transform([texts]) # apply TFIDF to text
    words = vectorizer.get_feature_names_out() # words/phrases from TFIDF
    scores = np.asarray(X.mean(axis=0))[0] # scores for above words/phrases from TFIDF
    embeddings = model.encode(words) # embed words into latent space
    normed = normalize(embeddings) # normalize embeddings
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=similarity_thresh) # set up clustering
    clustering.fit(normed) # cluster words into groups
    labels = clustering.labels_
    # combine words and scores from clusters
    df = None
    for n in range(clustering.n_clusters_): # for each cluster
        mask = (labels == n) # get all matching words in cluster
        text, summ = "", 0
        for word, score in zip(words[mask], scores[mask]): # for each word and score in cluster
            text += f"{word} " # smoosh
            summ += score # combine scores in cluster
        text = "/".join(set(text.split(" ")))
        if df is None:
            df = pd.DataFrame({"Idea":[text], "Score":[summ]}) # create dataframe
        else:
            df = pd.concat([df, pd.DataFrame({"Idea":[text], "Score":[summ]})]) # append to dataframe

    df = df.round(3)
    df.sort_values(by="Score", ascending=False) # sort dataframe by score
    return df

def save_to_csv(df:pd.DataFrame):
    print(os.getcwd()+"output.csv")
    df.to_csv(os.getcwd()+"/output.csv")

if __name__ == "__main__":
    with gr.Blocks() as demo:
        # "global" variables
        # embeddings = gr.State(None)
        pipe = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")
        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        with gr.Column() as outer_col:
            with gr.Row() as row1:
                grouping_strength = gr.Number(value=1.2, label="phrase grouping strength", maximum=2, minimum=0, precision=2, step=0.01)
                min_words = gr.Number(value=3, label="phrase length minimum", precision=0, minimum=1)
                max_words = gr.Number(value=5, label="phrase length maximum", precision=0, minimum=1)
            with gr.Row() as row2:
                with gr.Column() as col1:
                    files = gr.Files(label="images", file_types=["image"])
                    process_button1 = gr.Button(value="Process")
                    text_output = gr.Textbox(label="")
                    process_button1.click(fn=image_to_text, inputs=[files], outputs=[text_output])
                with gr.Column() as col2:
                    textbox = gr.Textbox(label="text to condense")
                    process_button2 = gr.Button(value="Process")
                    df = gr.DataFrame(wrap=True)
                    process_button2.click(fn=text_processing, inputs=[textbox, grouping_strength, min_words, max_words], outputs=[df])
                    save_to_csv_button = gr.Button(value="Save To CSV")
                    save_to_csv_button.click(fn=save_to_csv, inputs=df)
    demo.launch()
