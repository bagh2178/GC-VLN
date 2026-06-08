from habitat.config import Config
from pathlib import Path

CN = Config
# _C contains all the variables of the self-made models
_C = CN()

_C.UNI = CN()
_C.UNI.DEVICE = 'cuda'
_C.UNI.AGENT_HEIGHT = 0.88
_C.UNI.CAMERA_HEIGHT = 640
_C.UNI.CAMERA_WIDTH = 480
_C.UNI.MAX_RRAJ_LEN = 50
_C.UNI.VIDEO_OPTION = [] # options: "disk", "tensorboard"
_C.UNI.VIDEO_DIR = 'data/logs/video/'
_C.UNI.TELEFLAG = True
_C.UNI.THIN_TYPE = 1

path_to_sam2 = './Grounded-SAM-2'

_C.SCENEGRAPH = CN()
_C.SCENEGRAPH.TEXT_PROMPT = "chair."
_C.SCENEGRAPH.SAM2_CHECKPOINT = path_to_sam2 + "/checkpoints/sam2.1_hiera_large.pt"
# here can only be the reletive path
_C.SCENEGRAPH.SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_C.SCENEGRAPH.GROUNDING_DINO_CONFIG = path_to_sam2 + "/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_C.SCENEGRAPH.GROUNDING_DINO_CHECKPOINT = path_to_sam2 + "/gdino_checkpoints/groundingdino_swint_ogc.pth"
_C.SCENEGRAPH.BOX_THRESHOLD = 0.3
# useless in gsam server
_C.SCENEGRAPH.TEXT_THRESHOLD = 0.25
_C.SCENEGRAPH.DOORWAY_MAX_DEPTH_DIFF = 0.5
_C.SCENEGRAPH.VAR_THRESHOLD = 3

_C.MAP = CN()
_C.MAP.DEPTH_MAX = 10.0
_C.MAP.DEPTH_MIN = 0.1
_C.MAP.SCALE = 1
# eg: each grid cell = 5cm, the resolution = 100cm / 5cm
_C.MAP.RESOLUTION = 20
_C.MAP.AGENT_HEIGHT = _C.UNI.AGENT_HEIGHT * _C.MAP.RESOLUTION
_C.MAP.PASS_THRESHOLD = 5
_C.MAP.SIZE = 1000
_C.MAP.MASK_CAPTIONS = ['stair', 'stairs', 'stairstep', 'step', 'steps', 'staircase', 'staircases', 'stairsteps']

_C.RS = CN()
_C.RS.RADIUS = 15
_C.RS.DETECTAREA = 200
_C.RS.SAFEDISTANCE = 5
_C.RS.MAXDISTANCE = 20
_C.RS.TYPEWEIGHT = 2
_C.RS.DIS_INTERVAL = 7
_C.RS.NUM_DETECTIONS_THRESHOLD = 2
_C.RS.MIN_REGION_THRESHOLD = 150
_C.RS.RANDOM_EXP_TIME = 3
_C.RS.RANDOM_RADIUS = 30 # inner_radius is 5
_C.RS.EXP_THRESHOLD_ONE_STAGE = 2