import traceback
import sys
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.graph import copilot_app
from src.agent.prompts import SYSTEM_PROMPT_TEMPLATE

sys_msg = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(client_id='C-301', rm_tier='institutional'))
try:
    result = copilot_app.invoke({
        'messages': [sys_msg, HumanMessage(content='Search for exact guidelines on maximum equity allocation for conservative clients.')],
        'client_id': 'C-301',
        'rm_tier': 'institutional',
        'current_step': 0
    })
    print('AGENT RESULT:')
    print(result['messages'][-2].content)
    print(result['messages'][-1].content)
except Exception as e:
    print('CRASH:')
    traceback.print_exc()
