
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bitstring import BitArray


class utils:
    @staticmethod
    def is_hex(s):
        try:
            int(s, 16)
            return True
        except ValueError:
            return False


class Image():
    def __init__(self, left, top, width, height, packedField):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.packedField = packedField

    def getInfo(self):
            print("left ", self.left)
            print("top ", self.top)
            print("width ", self.width)
            print("height ", self.height)
            print("packedField ", self.packedField)

class GifFile:
    def __init__(self, fileName=""):
        self.index = 0
        if not (fileName.lower().endswith(".gif")):
            print("must be a gif file")
            return
        self.fileName = fileName
        if (self.fileName != ""):
            self.readFile()

    def readFile(self, fileName=""):
        if (fileName != ""):
            if not (fileName.lower().endswith(".gif")):
                print("must be a gif file")
                return
            self.fileName = fileName

        self.rawBytes = []
        f = open(self.fileName, "rb")
        byte = f.read(1)
        while byte:
            # Do stuff with byte.
            self.rawBytes.append(byte)
            byte = f.read(1)

        if(len(self.rawBytes) > 0):
            self.extractData()
        else:
            print("error cant read file as expected")

    def extractData(self):
        # header
        # All GIF files must start with a header block. The header takes up the first six bytes of the file.
        self.header = ""
        for i in range(0, 6):
            self.header = self.header + self.rawBytes[i].decode("ASCII")
        self.version = self.header[3:]

        # The logical screen descriptor always immediately follows the header.
        # This block tells the decoder how much room this image will take up. It is exactly seven bytes
        self.width = str(int.from_bytes(
            self.rawBytes[7], "little")) + str(int.from_bytes(self.rawBytes[6], "little"))
        self.height = str(int.from_bytes(
            self.rawBytes[9], "little")) + str(int.from_bytes(self.rawBytes[8], "little"))

        # The next byte contains four fields of packed data, the "logical screen descriptor"
        packedField = bin(int.from_bytes(
            self.rawBytes[10], "big")).lstrip('0b')
        print(self.rawBytes[10])
        print(packedField)

        # The first (most-significant) bit is the global color table flag. If it's 0,
        # then there is no global color table. If it's 1, then a global color table will follow.
        self.globalColorTableFlag = (packedField[0] == "1")

        # The next three bits are the color resolution.
        self.colorResolution = int(packedField[1:3], base=2)

        # The next single bit is the sort flag. If the values is 1,
        # then the colors in the global color table are sorted in order of "decreasing importance,"
        # which typically means "decreasing frequency" in the image.
        self.sortFlag = (packedField[4] == "1")

        # ignored?
        self.sizeOfGlobalColorTable = int(packedField[6:], base=2)

        # The next byte gives us the background color index. This byte is only meaningful if the global color table flag is 1
        self.backgroundColorIndex = int.from_bytes(
            self.rawBytes[11], "little")

        # The last byte of the logical screen descriptor is the pixel aspect ratio. Modern viewers don't use this
        self.pixelAspectRatio = int.from_bytes(
            self.rawBytes[13], "little")

        self.index = 13

        if self.globalColorTableFlag:
            self.readColorTable()

        self.readImages()

    def readColorTable(self):
        NumberOfEntries = 2**(self.colorResolution+1)
        byteSize = 3*NumberOfEntries

        self.globalColorTable = []

        i = self.index
        while i < self.index+byteSize:
            self.globalColorTable.append((int.from_bytes(self.rawBytes[i], "little"), int.from_bytes(
                self.rawBytes[i+1], "little"), int.from_bytes(self.rawBytes[i+2], "little")))
            i += 3

        self.index = self.index+byteSize

    def readImages(self):
        self.images = []
        localIndex = 0
        for byte in self.rawBytes:
            if localIndex < self.index:
                pass
            else:
                # Every image descriptor begins with the value 2C. The next 8 bytes represent the location and size of the following image.
                if byte.hex() == "2c":
                    print("index1",localIndex)
                    self.readImage(localIndex)

            localIndex += 1

    def readImage(self, index):
        print("index2",index)

        # Each image begins with an image descriptor block. This block is exactly 10 bytes long.
        left = str(int.from_bytes(
            self.rawBytes[index+2], "little")) + str(int.from_bytes(self.rawBytes[index+1], "little"))
        top = str(int.from_bytes(
            self.rawBytes[index+4], "little")) + str(int.from_bytes(self.rawBytes[index+3], "little"))
        width = str(int.from_bytes(
            self.rawBytes[index+6], "little")) + str(int.from_bytes(self.rawBytes[index+5], "little"))
        height = str(int.from_bytes(
            self.rawBytes[index+8], "little")) + str(int.from_bytes(self.rawBytes[index+7], "little"))
        packedField = bin(int.from_bytes(
            self.rawBytes[index+9], "little")).lstrip('0b')

        self.images.append(Image(
            left,
            top,
            width,
            height,
            packedField
        ))

    def showImagesInfo(self):
        localIndex = 0
        print("Images Info")
        for image in self.images:
            print("data from image ",localIndex)
            image.getInfo()
            localIndex += 1
            print("-----------------")
        print("\n")

    def getBasicInfo(self):
        print("GIF INFO")
        print("header ", self.header)
        print("version ", self.version)
        print("width ", self.width)
        print("height ", self.height)
        print("globalColorTableFlag ", self.globalColorTableFlag)
        print("colorResolution ", self.colorResolution)
        print("sizeOfGlobalColorTable ", self.sizeOfGlobalColorTable)
        print("backgroundColorIndex ", self.backgroundColorIndex)
        print("pixelAspectRatio ", self.pixelAspectRatio)
        print("-------------------------\n")

    def getPixelInfo(self, x, y):
        pass


a = GifFile("sample_1.gif")
a.getBasicInfo()
a.showImagesInfo()
