import gradio as gr
from sidekick import Sidekick


async def setup():
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick


async def user_message(message, history):
    return "", history + [{"role": "user", "content": message}]


async def process_message(sidekick, success_criteria, history):
    # The message here is the last one in history
    actual_message = history[-1]["content"]
    # Pass history[:-1] to sidekick to avoid double counting the user message we just added
    results = await sidekick.run_superstep(actual_message, success_criteria, history[:-1])
    return results, sidekick


async def reset():
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", None, new_sidekick


def free_resources(sidekick):
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## Sidekick Personal Co-Worker")
    sidekick = gr.State(delete_callback=free_resources)

    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=600, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False, placeholder="What are your success critiera?"
            )
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    ui.load(setup, [], [sidekick])
    
    # Chain events for immediate feedback
    message.submit(
        user_message, [message, chatbot], [message, chatbot], queue=False
    ).then(
        process_message, [sidekick, success_criteria, chatbot], [chatbot, sidekick]
    )
    
    success_criteria.submit(
        user_message, [message, chatbot], [message, chatbot], queue=False
    ).then(
        process_message, [sidekick, success_criteria, chatbot], [chatbot, sidekick]
    )
    
    go_button.click(
        user_message, [message, chatbot], [message, chatbot], queue=False
    ).then(
        process_message, [sidekick, success_criteria, chatbot], [chatbot, sidekick]
    )
    
    reset_button.click(reset, [], [message, success_criteria, chatbot, sidekick])


ui.launch(inbrowser=True)
