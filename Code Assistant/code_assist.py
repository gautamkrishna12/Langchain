import requests
import json
import gradio as gr

app_url="http://localhost:11434/api/generate"

headers={
    'Content-Type':'application/json'
}

model_history=[]

def Get_Model_Response(query):
    model_history.append(query)
    query_with_history="\n".join(model_history)

    data={"model":"CodeAi","prompt":query_with_history,"stream":False}

    response=requests.post(app_url,headers=headers,data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        return data['response']
    else:
        print(f'error:{response.text}')

Interface=gr.Interface(
    fn=Get_Model_Response,
    inputs=gr.Textbox(lines=10,placeholder="Enter your Prompt"),
    #outputs='text'
    outputs=gr.Textbox(
        lines=10,                      
        label="Model Response"
    )
)
Interface.launch(share=True)
    



