# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:47:57 2024

@author: Daniel Pessoa
"""

import pyautogui
import time

def press_keys():
    pyautogui.press('right')  # Pressiona a tecla direita
    time.sleep(1)  # Aguarda 1 segundo entre as teclas
    pyautogui.press('left')  # Pressiona a tecla esquerda
    time.sleep(1)  # Aguarda 1 segundo entre as teclas
    pyautogui.press('up')  # Pressiona a tecla esquerda
    time.sleep(1)  # Aguarda 1 segundo entre as teclas
    pyautogui.press('down')  # Pressiona a tecla esquerda
    print('Tecla pressionada')

if __name__ == "__main__":
    try:
        while True:
            press_keys()
            time.sleep(30)  # Aguarda 2 minutos antes de repetir
    except KeyboardInterrupt:
        print("\nScript interrompido pelo usuário.")