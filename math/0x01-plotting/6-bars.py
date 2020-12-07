#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
applebar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[0], width=.5,
                   color='red', label='Apples')
bananabar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[1], width=.5,
                    color='yellow', label='Bananas', bottom=fruit[0])
orangebar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[2],
                    color='#ff8000', label='Oranges', width=.5,
                    bottom=fruit[0] + fruit[1])
peachbar = plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[3],
                   color='#ffe5b4', label='Peaches', width=.5,
                   bottom=fruit[0] + fruit[1] + fruit[2])
plt.legend(handles=[applebar, bananabar, orangebar, peachbar])
plt.show()
