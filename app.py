import gradio as gr
from easyocr import Reader
from PIL import Image
import io
import json
import csv
import openai
import ast
import os
from deta import Deta


######################
import requests
import json

import os
import openai



class OpenAI_API:
    def __init__(self):
        self.openai_api_key = ''
        
    def single_request(self, address_text):
        
        openai.api_type = "azure"
        openai.api_base = "https://damlaopenai.openai.azure.com/"
        openai.api_version = "2022-12-01"
        openai.api_key = os.getenv("API_KEY")
        
        response = openai.Completion.create(
          engine="Davinci-003",
          prompt=address_text,
          temperature=0.9,
          max_tokens=256,
          top_p=1.0,
          n=1,
          logprobs=0,
          echo=False,
          stop=None,
          frequency_penalty=0,
          presence_penalty=0,
          best_of=1)

        return response

########################

openai.api_key = os.getenv('API_KEY')
reader = Reader(["tr"])


def get_parsed_address(input_img):

    address_full_text = get_text(input_img)
    return openai_response(address_full_text)


def preprocess_img(inp_image):
    gray = cv2.cvtColor(inp_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bitwise_not(gray)
    return gray_img


def get_text(input_img):
    result = reader.readtext(input_img, detail=0)
    return " ".join(result)


def save_csv(mahalle, il, sokak, apartman):
    adres_full = [mahalle, il, sokak, apartman]

    with open("adress_book.csv", "a", encoding="utf-8") as f:
        write = csv.writer(f)
        write.writerow(adres_full)
    return adres_full


def get_json(mahalle, il, sokak, apartman):
    adres = {"mahalle": mahalle, "il": il, "sokak": sokak, "apartman": apartman}
    dump = json.dumps(adres, indent=4, ensure_ascii=False)
    return dump

def write_db(data_dict):
    # 2) initialize with a project key
    deta_key = os.getenv('DETA_KEY')
    deta = Deta(deta_key)

    # 3) create and use as many DBs as you want!
    users = deta.Base("deprem-ocr")
    users.insert(data_dict)


def text_dict(input):
    eval_result = ast.literal_eval(input)
    write_db(eval_result)

    return (
        str(eval_result['city']),
        str(eval_result['distinct']),
        str(eval_result['neighbourhood']),
        str(eval_result['street']),
        str(eval_result['address']),
        str(eval_result['tel']),
        str(eval_result['name_surname']),
        str(eval_result['no']),
    )
        
def openai_response(ocr_input):
    prompt = f"""Tabular Data Extraction You are a highly intelligent and accurate tabular data extractor from 
            plain text input and especially from emergency text that carries address information, your inputs can be text 
            of arbitrary size, but the output should be in [{{'tabular': {{'entity_type': 'entity'}} }}] JSON format Force it 
            to only extract keys that are shared as an example in the examples section, if a key value is not found in the 
            text input, then it should be ignored. Have only city, distinct, neighbourhood, 
            street, no, tel, name_surname, address Examples: Input: Deprem sırasında evimizde yer alan adresimiz: İstanbul, 
            Beşiktaş, Yıldız Mahallesi, Cumhuriyet Caddesi No: 35, cep telefonu numaram 5551231256, adim Ahmet Yilmaz 
            Output: {{'city': 'İstanbul', 'distinct': 'Beşiktaş', 'neighbourhood': 'Yıldız Mahallesi', 'street': 'Cumhuriyet Caddesi', 'no': '35', 'tel': '5551231256', 'name_surname': 'Ahmet Yılmaz', 'address': 'İstanbul, Beşiktaş, Yıldız Mahallesi, Cumhuriyet Caddesi No: 35'}}
            Input: {ocr_input}
            Output:
        """

    openai_client = OpenAI_API()
    response = openai_client.single_request(ocr_input)
    resp = response["choices"][0]["text"]
    print(resp)
    resp = eval(resp.replace("'{", "{").replace("}'", "}"))
    resp["input"] = ocr_input
    dict_keys = [
    'city',
    'distinct',
    'neighbourhood',
    'street',
    'no',
    'tel',
    'name_surname',
    'address',
    'input',
    ]
    for key in dict_keys:
        if key not in resp.keys():
            resp[key] = ''
    return resp


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Enkaz Bildirme Uygulaması
    """)
    gr.Markdown("Bu uygulamada ekran görüntüsü sürükleyip bırakarak AFAD'a enkaz bildirimi yapabilirsiniz. Mesajı metin olarak da girebilirsiniz, tam adresi ayrıştırıp döndürür. API olarak kullanmak isterseniz sayfanın en altında use via api'ya tıklayın.")
    with gr.Row():
        img_area = gr.Image(label="Ekran Görüntüsü yükleyin 👇")
        ocr_result = gr.Textbox(label="Metin yükleyin 👇 ")
    open_api_text = gr.Textbox(label="Tam Adres")
    submit_button = gr.Button(label="Yükle")
    with gr.Column():
        with gr.Row():
            city = gr.Textbox(label="İl")
            distinct = gr.Textbox(label="İlçe")
        with gr.Row():
            neighbourhood = gr.Textbox(label="Mahalle")
            street = gr.Textbox(label="Sokak/Cadde/Bulvar")
        with gr.Row():
            tel = gr.Textbox(label="Telefon")
        with gr.Row():
            name_surname = gr.Textbox(label="İsim Soyisim")
            address = gr.Textbox(label="Adres")
        with gr.Row():
            no = gr.Textbox(label="Kapı No")


    submit_button.click(get_parsed_address, inputs = img_area, outputs = open_api_text, api_name="upload_image")

    ocr_result.change(openai_response, ocr_result, open_api_text, api_name="upload-text")

    open_api_text.change(text_dict, open_api_text, [city, distinct, neighbourhood, street, address, tel, name_surname, no])


if __name__ == "__main__":
    demo.launch()