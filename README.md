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
- A realtively (58% test accuracy) accurate ticket classifier using PyTorch to identify lottery ticket types from images.

## Things to add
- A better ticket classifier with higher accuracy.
- A number extractor using OCR (Optical Character Recognition) to read numbers from ticket images.
- User interface for easier interaction.
- Support for more lottery games

