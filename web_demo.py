from transformers import AutoModel, AutoTokenizer
import gradio as gr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tokenizer = AutoTokenizer.from_pretrained("./weights/adgen-chatglm-6b-pt-8-1e-2/checkpoint-30",
                                         trust_remote_code=True, revision='')
model = AutoModel.from_pretrained("./weights/adgen-chatglm-6b-pt-8-1e-2/checkpoint-30",
                                  trust_remote_code=True, revision='').half().cuda()

model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="User：" + query))
            updates.append(gr.update(visible=True, value="HouYi model(后羿大模型)：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates


with gr.Blocks() as demo:
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="提问："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="回复："))

    with gr.Row():        
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(
                container=False)
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            button = gr.Button("Generate")
    gr.Markdown('Welcome to HouYi Model 欢迎使用后羿大模型')
    button.click(predict, [txt, max_length, top_p, temperature, state], [state] + text_boxes)
demo.queue().launch(share=False, server_name='192.168.1.100', inbrowser=True)  # https://zhuanlan.zhihu.com/p/556126344
