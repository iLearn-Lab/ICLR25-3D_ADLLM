# from .openad_pn2_modify_demo import OpenAD_PN2_modify_demo
# from .openad_pn2_modify_sam_decoder_demo import OpenAD_PN2_modify_samdecoder_demo
from .openad_dgcnn_modify import OpenAD_DGCNN_modify
# from .openad_dgcnn_modify_demo import OpenAD_DGCNN_Modify_demo
from .openad_pn2_modify import OpenAD_PN2_modify
from .openad_pn2_modify_sam_decoder import OpenAD_PN2_modify_samdecoder
from .model_PT_512_MLP import PointTransformerSeg_512MLP
__all__ = [
    # model for extract feature
    'OpenAD_DGCNN_modify',
    'OpenAD_PN2_modify',
    'OpenAD_PN2_modify_samdecoder',
    # Demo
    # 'OpenAD_PN2_modify_demo',
    # 'OpenAD_PN2_modify_samdecoder_demo',
    # 'OpenAD_DGCNN_Modify_demo',
    'PointTransformerSeg_512MLP'
    ]