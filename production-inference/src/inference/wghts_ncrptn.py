from pathlib import Path
import argparse

class Guardian:
    
    def __init__(self, weights: str):
        self.weights = Path(weights)
        with open(weights, 'rb') as f:
            self.weights_binary = bytearray(f.read())
        return None
    
    def encode(self) -> bytes:
        i, j = 0, len(self.weights_binary) - 1
        while i < j:
            self.weights_binary[i], self.weights_binary[j] = self.weights_binary[j], self.weights_binary[i]
            i += 1
            j -= 1
        return bytes(self.weights_binary)
    
    def decode(self) -> bytes:
        i, j = 0, len(self.weights_binary) - 1
        while i < j:
            self.weights_binary[i], self.weights_binary[j] = self.weights_binary[j], self.weights_binary[i]
            i += 1
            j -= 1
        return bytes(self.weights_binary)
    
    def save_guarded_weights(self, dirname: str) -> None:
        dirname = Path(dirname)
        if not dirname.exists():
            dirname.mkdir(parents=True)

        arr = self.encode()
        filename = dirname / (f'{self.weights.stem}-grdd{self.weights.suffix}')
        with open(Path(filename), 'wb') as file:
            file.write(arr)
        return None


def main():
    EXTENSIONS = {'.pt', '.onnx', '.trt'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='model path or dirname to decode')
    parser.add_argument('--dirname2save', type=str, default='.', help='The name of the folder where to save to')
    opt = parser.parse_args()
    
    weights = Path(opt.weights)
    if weights.is_file():
        weights = [weights]
    else:
        buffer = []
        for w in weights.iterdir():
            if not w.suffix in EXTENSIONS:
                continue
            buffer.append(w)
        weights = buffer
    
    for w in weights:
        guardian = Guardian(w)
        guardian.save_guarded_weights(opt.dirname2save)
    return None



if __name__ == '__main__':
    main()