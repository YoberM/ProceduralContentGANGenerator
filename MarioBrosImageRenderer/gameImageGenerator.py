from PIL import Image,ImageDraw
from typing import Dict, List, Tuple
import sys
import os
import numpy as np

class GameImageGenerator:
    def __init__(self, asset_map: Dict[int, Image.Image]):
        """
        Class to facilitate the rendering of a game level from given feature map.
        
        Args:
            asset_map (Dict[int, Image]): Used to map feature->image_asset.
        """
        self.asset_map = asset_map
        self.level = None

    def render(self, image_array: np.array, sprite_dims: Tuple, bg_color="white"):
        """
        Renders the image given an image array.
        Assumes that array is 2D i.e only one channel,
        with an integer associated to a certain asset/sprite.
        """

        level_dims = (
            image_array.shape[1] * sprite_dims[1],
            image_array.shape[0] * sprite_dims[0],
        )  # x,y
        self.level = Image.new(mode="RGBA", size=level_dims, color=bg_color)

        for i, row in enumerate(image_array):
            for j, asset_id in enumerate(row):
                if asset_id not in self.asset_map:
                    continue
                self.level.paste(
                    self.asset_map[asset_id],
                    (sprite_dims[0] * j, sprite_dims[0] * i),
                    self.asset_map[asset_id],
                )

        # self.level.show()
        # self.level.save(img_name + ".png")
        return self.level

    def save_gen_level(self, img_name="trial_image"):
        self.level.save(image_name + ".png")


class TextToImageGenerator(GameImageGenerator):

    def __init__(self, game, tile_size, tile_path, bg_color='white'):
        self.tile_path, self.symbols = self.symbolMap(game, tile_path)
        self.gameImage = None
        self.tile_size = (tile_size, tile_size)
        self.bg_color = bg_color
        super().__init__(asset_map=self.setAssetMap())

    def symbolMap(self, game, tile_path):
        symbol_arr = []
        if game == 'mario_ai':
            tile_path += '/marioAITiles/{}.png'
            symbol_arr = ['X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']', 'o', 'B', 'b']
        
        elif game == 'test_mario':
            tile_path += '/testSymbols/{}.png'
            symbol_arr = ["=","e","=","c","c","f","p","?","c","#","-","p","p","p","#","?","=","=","e","p","p","p","p","p","p","-"]

        elif game == 'lode_runner':
            opFile = open(tile_path+'/lode_runner_tiles/textToImage.txt', 'r')
            tile_path += '/lode_runner_tiles/{}.png'
            symbol_arr = list(self.inputText(opFile).flatten())

        else:
            ValueError()

        return tile_path, symbol_arr


    def setAssetMap(self):
        asset_map = {}

        for i in range(len(self.symbols)):
            try:
                asset_map[i] = Image.open(self.tile_path.format(i)).convert("RGBA")
            except FileNotFoundError:
                continue

        return asset_map

    def inputText(self, file):
        readLines = []
        while True:
            line = file.readline()
            if not line:
                break
            readLines.append([c for c in line if c!='\n'])
        return np.array(readLines)

    def readTextLevel(self, textLevelPath, textLevelFileName):
        textFile = open(textLevelPath+os.sep+textLevelFileName, "r")
        
        symbol_matrix = self.inputText(textFile)
        index_matrix = np.empty(shape=symbol_matrix.shape, dtype=int)

        for i in range(symbol_matrix.shape[0]):
            for j in range(symbol_matrix.shape[1]):
                if(symbol_matrix[i][j] != '\n'):
                    index_matrix[i][j] = self.symbols.index(symbol_matrix[i][j])
        textFile.close()

        return index_matrix

    def viewLevel(self, textLevelPath, textLevelFileName):
        indexMatrix = self.readTextLevel(textLevelPath, textLevelFileName)
        levelImage = self.render(indexMatrix, self.tile_size, self.bg_color)
        self.gameImage = levelImage

    def saveLevelImage(self, path, imageName):
        self.gameImage.save(path+os.sep+os.path.splitext(imageName)[0]+'.png')




if __name__ == "__main__":
    print("Running main from image generator")
    project_path = 'MarioBrosImageRenderer/'
    level_path = project_path + 'GeneratedLevels' #'GameImageGenerator/Lode Runner/Processed'#'DagstuhlGAN-master/marioaiDagstuhl/data/mario2J/Processed'
    level_files = []# 'SuperMarioBros2(J)-World1-1.txt'
    tile_path = project_path + 'tileImages'
    output_path = project_path + 'GeneratedLevelsImages'
    tile_size = 16 # CHANGE: for lode runner it is 8; mario is 16
    
    bg_color = '#00b3ff'
    view_game = TextToImageGenerator('mario_ai', tile_size, tile_path, bg_color=bg_color)


    files = []
    path = "data/"
    labels = []
    for textImage in os.listdir(level_path):
        # Escribir la información en el CSV
        level_files.append(textImage)

    for level_file in level_files:
        view_game.viewLevel(level_path, level_file)
        view_game.saveLevelImage(output_path, level_file)

