
def ctrl_hair_parser_options(parser):
    parser.add_argument('-c', '--config', type=str, default='001')
    parser.add_argument('-g', '--gpu', type=str, help='Specify GPU number', default='0')
    parser.add_argument('-n', '--need_crop', type=bool, help='whether images need crop', default=True)
    parser.add_argument('--no_blending', action='store_true',
                        help='whether using poisson blending as post processing', default=False)