# Lottery-Application

This repository contains a Python application that interacts with lottery APIs to fetch and analyze lottery data. It includes functionalities to retrieve datasets for different lottery games, process the data, and check for winning numbers.

The goal of this project is to allow users to take a photo of their lottery ticket numbers and the program will check if they have won any prizes based on the latest lottery draws.

This will be done using image recognition techniques to determine lottery ticket type, and extract the numbers from the photo and then comparing them against the fetched lottery data.

## Features
- Fetch lottery data from public APIs.
- Process and store lottery datasets in CSV format.
- Analyze lottery numbers to check for wins based on user input.
- This has support for the following lottery games:
  - Powerball
  - Mega Millions
  - Lotto America
  - EuroMillions
- A realtively (89% test accuracy with dataset images) accurate ticket classifier using PyTorch to identify lottery ticket types from images.
- A perspective correction module using OpenCV to correct the perspective of the lottery ticket images before performing OCR (still needing to be implemented).
- Can get the date of the lottery draw from OCR results
- Getting the numbers from the lottery ticket using OCR (currently has some reliability issues that are being slowly fixed).


## Currently Developing
- Creating a terminal-based user interface to create core app functionality, then will move on to a more user-friendly interface.

## Things to add
- A better ticket classifier with higher accuracy.
- User interface for easier interaction.
- Support for more lottery games
- A more accurate way of getting the corners of the lottery ticket for perspective warping. 


