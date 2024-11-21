import os


BASE_DIR = os.path.expanduser("~/Documents/")
if not os.path.exists(BASE_DIR):
    raise Warning(f"{BASE_DIR} does not exist. check your environment.")

def create_and_move(name:str)->str|None:
    '''
    cereate temp dir and move to it.
    return full path of the dir
    '''
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        os.makedirs(path)
        os.chdir(path)
        return path
    
    result = input(f"{path} already exists. want to overwrite?[y/n]")
    if result.lower() != 'y':
        return None
    
    os.makedirs(path)
    os.chdir(path)
    return path





# if __name__ == "__main__":
#     create_and_move("A")
#     print(os.getcwd())
