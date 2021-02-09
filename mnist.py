import sys
import numpy as np
import pandas as pd
import pygame.font, pygame.event, pygame.draw
from NeuralNetwork import NeuralNetwork

try:
    df = pd.read_csv("dataset_mnist.csv")
except:
    exit("Error: Something went wrong with the dataset")
tmp = df["label"].tolist()
Y = np.zeros((len(tmp), 10))
for i,x in enumerate(tmp):
    Y[i][x] = 1

X = df.drop(columns=["label"]).to_numpy()

model = NeuralNetwork()

model.add_layer(200, input_dim=X.shape[1], activation='sigmoid')
model.add_layer(100, activation='sigmoid')
model.add_layer(50, activation='sigmoid')
model.add_layer(10, activation='softmax')


model.fit(X, Y, epoch=100, verbose=1, normalize=True, lr=0.9)

pygame.init()

BLACK = "#000000"
WHITE = "#FFFFFF"
COLOR1 = "#A8E6CE"
COLOR2 = "#DCEDC2"
COLOR3 = "#FFD3B5"
COLOR4 = "#FFAAA6"
COLOR5 = "#FF8C94"

width, height = 412, 512
size = [width, height]
input_field = (392, 392)
edge_buffer = (10, 10)

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Model trained on the MNIST database")
background = pygame.Surface(input_field)
background.fill(WHITE)
font_small = pygame.font.SysFont("Agency FB", 32)

def display_prediction(prediction):

    font = pygame.font.SysFont("Agency FB", 36)
    initialize_prediction = font.render("Prediction: %s" %(prediction), 1, (WHITE))
    pygame.draw.rect(screen, COLOR4, (edge_buffer[0], input_field[1] + edge_buffer[1] + 10, input_field[0], 90))
    screen.blit(initialize_prediction, (edge_buffer[0] + 5, input_field[1] + edge_buffer[1] + 20))

def draw_line(start, end):

    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0]+i/distance*dx)
        y = int(start[1]+i/distance*dy)
        pygame.draw.circle(background, BLACK, (x - edge_buffer[0], y - edge_buffer[1]), 9)

def create_button(btn_label, locationX, locationY, width, height):

    initialize_btn_label = font_small.render(btn_label, 1, WHITE)
    initialize_btn_label_dim = initialize_btn_label.get_rect().width, initialize_btn_label.get_rect().height
    pygame.draw.rect(screen, COLOR2, (locationX, locationY, width, height))
    mouse = pygame.mouse.get_pos()
    if locationX + width > mouse[0] > locationX and locationY + height > mouse[1] > locationY:
        pygame.draw.rect(screen, COLOR3, (locationX, locationY, width, height))
        if pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Classify":
            calculate_prediction(background)
        elif pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Clear":
            background.fill(WHITE)
            display_prediction('Unknown')
    else:
        pygame.draw.rect(screen, COLOR2, (locationX, locationY, width, height))
    screen.blit(initialize_btn_label, (locationX + width/2 - initialize_btn_label_dim[0]/2,
        locationY + height/2 - initialize_btn_label_dim[1]/2))


def calculate_prediction(background):
    scaledBackground = pygame.transform.smoothscale(background, (28, 28))
    image = pygame.surfarray.array3d(scaledBackground)
    image = 253 - image
    image = np.mean(image, 2)
    image = image.transpose()
    image = image.ravel()
    reshaped_input = image.reshape(1,784)
    prediction = model.predict(reshaped_input)
    pred = np.argmax(prediction)
    display_prediction(pred)

def main():

    last_pos = (0, 0)
    screen.fill(COLOR1)
    display_prediction('Unknown')
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    draw_line(event.pos, last_pos)
                last_pos = event.pos
            screen.blit(background, (edge_buffer[0], edge_buffer[1]))
            create_button("Classify", input_field[0] - 110, input_field[1] + edge_buffer[1] + 10, 120, 45)
            create_button("Clear", input_field[0] - 110, input_field[1] + edge_buffer[1] + 55, 120, 45)
            pygame.display.flip()

if __name__ == "__main__":
    main()