from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self._parser.add_argument('--face_aus_path', type=str, default='./face_expression.pkl', help='face aus path')
        self.is_train = False
