def validate_output(text:str):
    banned = ['guaranteed return','100% safe']
    for b in banned:
        if b in text.lower():
            return False
    return True
