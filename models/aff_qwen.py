import logging
from typing import List
import os
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from common.registry import registry
from common.utils import get_rank
from easydict import EasyDict
from models.base_model import BaseModel, disabled_train
from .AFD import build_AFD
from models.AFD.aff_utils.loss import dice_loss, sigmoid_ce_loss
from models.pointbert.point_encoder import PointTransformer
from models.openad.utils import build_model_checkpointfromddp
from gorilla.config import Config
from models.utils import Modify_cfg_from_yaml_file

SEG_TOKEN = "<SEG>"


@registry.register_model("aff_qwen")
class AffordanceQwen(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "qwen": "/workspace/project/Research_3D_Aff/Programme_affllm_code_chs/configs/models/point_qwen.yaml",
    }

    def __init__(
        self,
        mix_precision="bf16",
        # The Seg point feature dimension
        prompt_encoder_dim=32,
        # Loss
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        uselabelratio=False,
        # Point_encoder
        point_model_config_path=None,
        freeze_point=True,
        # Seg_Pointencoder(PointBackbone)
        free_seg_point_encoder=True,
        seg_point_encoder_config_path=None,
        seg_point_encoder_path=None,
        # AFD
        aff_path=None,
        train_aff_decoder=False,
        upscale_points=2048,
        # Lora
        lora_r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        # LLM
        llm_model=None,
        freeze_llm=True,
        prompt="",
        max_txt_len=256,
        max_output_txt_len=128,
        freeze_linear=True,
        label_ratio_path="/workspace/project/Research_3D_Aff/Programme_affllm_code_chs/result_ratio.json",
        lora_llm_finetune=False,
        **kwargs,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        # Set Precision
        self.mix_precision = mix_precision
        # Point_Encoder，ULIP2_PointBert initlize Point_encoder
        self.point_model_config = Modify_cfg_from_yaml_file(point_model_config_path)
        self.point_encoder = PointTransformer(self.point_model_config.model)
        self.point_encoder.load_checkpoint(self.point_model_config.model_path)

        # The PointBackbone
        openadcfg = Config.fromfile(seg_point_encoder_config_path)
        self.seg_point_encoder = build_model_checkpointfromddp(
            openadcfg, seg_point_encoder_path, is_eval=False
        )
        self.upscale_points = upscale_points

        # Loss Weight
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.uselabelratio = uselabelratio

        # Label Ratio
        if self.uselabelratio:
            with open(label_ratio_path, "r", encoding="utf-8") as ratio_file:
                self.label_ratio = json.load(ratio_file)

        logging.info(f"语言ce_loss_weight:{self.ce_loss_weight}")
        logging.info(f"Mask—-dice_loss_weight:{self.dice_loss_weight}")
        logging.info(f"Mask--bce_loss_weight:{self.bce_loss_weight}")
        logging.info(f"Use_label_ratio:{self.uselabelratio}")

        # The PointEncoder before the LLM
        if freeze_point:
            for name, param in self.point_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.point_encoder.eval()
            self.point_encoder.train = disabled_train
            print("Freeze point encoder")
            logging.info("Freeze point encoder")

        # The PointBackbone Before AFD
        if free_seg_point_encoder:
            for name, param in self.seg_point_encoder.named_parameters():
                param.requires_grad = False
            self.seg_point_encoder = self.seg_point_encoder.eval()
            self.seg_point_encoder.train = disabled_train
            print("Freeze Seg_point encoder (PointBackbone)")
            logging.info("Freeze Seg_point encoder (PointBackbone)")

        # LLM
        print("Start Load LLM")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, use_fast=False, trust_remote_code=True, truncation_side="left"
        )
        print("llm_tokenizer pad_token", self.llm_tokenizer.pad_token)
        print("llm_tokenizer pad_token", self.llm_tokenizer.pad_token_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float32
        )
        print("Load LLM Model Successfully")
        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            print("Freeze LLM")
            logging.info("Freeze LLM")

        # Add Lora
        if lora_llm_finetune:
            print("Using Lora LLM Finetune")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()

        # Set AFF token
        self.llm_tokenizer.add_tokens([SEG_TOKEN])
        self.seg_token_id = self.llm_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        self.llm_model.base_model.model.model.embed_tokens.weight.requires_grad = True
        print("Embed_token Trainable!")
        print("AFF_token_ID", self.seg_token_id)
        logging.info(f"AFF_token_ID:{self.seg_token_id}")

        # points to llm projector
        self.llm_proj = nn.Linear(
            self.point_model_config.model.trans_dim, self.llm_model.config.hidden_size
        )
        if freeze_linear:
            for name, param in self.llm_proj.named_parameters():
                param.requires_grad = False
            self.llm_proj = self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            print("freeze point encoder to LLM liner")

        # Pointcloud to llm projector
        self.llm_proj = nn.Linear(
            self.point_model_config.model.trans_dim, self.llm_model.config.hidden_size
        )
        if freeze_linear:
            for name, param in self.llm_proj.named_parameters():
                param.requires_grad = False
            self.llm_proj = self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            print("freeze point encoder to LLM liner")

        # Initialize the AFD Module
        self.prompt_encoder_dim = prompt_encoder_dim
        self.aff_model, self.aff_proj = self.initialize_affordance_modules(
            aff_path,
            in_dim=self.llm_model.config.hidden_size,
            out_dim=self.prompt_encoder_dim,
            train_aff_decoder=train_aff_decoder,
        )
        # #Calculate the total parameters number of aff_model
        # print("aff_model参数量")
        # self.aff_model.counting_training_parameters()
        # # Calculate the total number of aff_proj trainable parameters
        # aff_proj_total_params = sum(p.numel() for p in self.aff_proj.parameters() if p.requires_grad)
        # print(f"The aff_projection layer has a total of {aff_proj_total_params} trainable parameters.")

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        self.counting_training_parameters()

    # The Function of Initializing AFD Module
    def initialize_affordance_modules(
        self, aff_path, in_dim, out_dim, train_aff_decoder=True
    ):
        aff_model = build_AFD(aff_path)
        # AFD Module
        if train_aff_decoder:
            aff_model.train()
            for param in aff_model.parameters():
                param.requires_grad = True
        # Projection layer
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        aff_proj = nn.Sequential(*text_fc)
        # aff proj
        aff_proj.train()
        for param in aff_proj.parameters():
            param.requires_grad = True
        return aff_model, aff_proj

    # Count Training Parameters
    def counting_training_parameters(self):
        total = 0.0
        trainable_names = []
        all = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.nelement()
                trainable_names.append(name)
            all += param.nelement()
        print(trainable_names)
        print("  + Number of trainable params: %.2fM" % (total / 1e6))
        print("Number of all params: %.2fM" % (all / 1e6))

    # The Point Encoder before PointMLLM
    def encode_point(self, points):
        with self.maybe_autocast(self.mix_precision):
            # The input data is a list
            points2bs = torch.stack(points)
            points_feat, points_pos = self.point_encoder(points2bs)
            inputs_llm = self.llm_proj(points_feat)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            points2bs.device
        )
        return inputs_llm, atts_llm, points_pos

    # Predict Affordance Mask
    def predict_mask(
        self,
        output_ids,
        last_hidden_states,
        points,
        shape_id,
        original_list=(1, 2048),
    ):
        original_list = (1, self.upscale_points)
        if points is None:
            return []
        # Obtain the Dense pointcloud embeddings from seg point encoder(PointBackbone)
        points2bs = torch.stack(points)
        points2bs = points2bs.transpose(1, 2)
        point_embedding = self.seg_point_encoder(points2bs)
        projected_hidden_state = self.aff_proj(
            last_hidden_states
        )  # [bs,length,dim]----->10*6*1536-->10*6*32
        pred_masks = []
        for i in range(projected_hidden_state.shape[0]):
            seg_token_mask = output_ids[i][:] == self.seg_token_id
            if seg_token_mask.sum() == 0:
                pred_masks.append(
                    torch.zeros((1, *original_list), dtype=torch.float32).cuda()
                )
                continue
            seq_length = last_hidden_states.shape[1]
            seg_token_mask = seg_token_mask[:seq_length]
            pred_embeddings_ = projected_hidden_state[i][seg_token_mask]
            point_embedding_ = point_embedding[i].unsqueeze(0)
            pred_mask = self.aff_model(
                pointcloud_embeddings=point_embedding_,
                pointcloud_emorigin=point_embedding_,
                sparse_prompt_embeddings=pred_embeddings_.unsqueeze(1),
                multimask_output=False,
            )
            pred_masks.append(pred_mask)
        return pred_masks

    # Prepare Input for LLM
    def prepare_input(self, question):
        instruction = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return instruction

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(
                torch.cat(
                    [
                        input_ids[i][:this_input_ones],
                        output_ids[i][0:],
                        input_ids[i][this_input_ones:],
                    ]
                )
            )
            llm_tokens["attention_mask"].append(
                torch.cat(
                    [
                        input_atts[i][:this_input_ones],
                        output_atts[i][0:],
                        input_atts[i][this_input_ones:],
                    ]
                )
            )
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    # Prepare Input for LLM

    # Forward
    def forward(self, samples):
        questions = samples["question"]
        answers = samples["answer"]
        bs = len(questions)
        new_questions = []
        for i in range(bs):
            res = self.prepare_input(questions[i])
            new_questions.append(res)
        questions = new_questions
        new_answers = []
        for i in range(bs):
            res = self.prepare_response(answers[i])
            new_answers.append(res)
        answers = new_answers

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(self.device)

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in answers],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            add_special_tokens=False,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        attention_mask = llm_tokens["attention_mask"]

        with self.maybe_autocast(self.mix_precision):
            if "points" in samples:
                points = samples["points"]

                inputs_llm, atts_llm, point_pos_em = self.encode_point(points)
                # do not apply loss to the query tokens
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long)
                    .to(inputs_llm.device)
                    .fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True,
            )
            # predict masks
            points = samples.get("points", None)
            gt_masks = samples.get("masks", None)
            shape_id = samples.get("shape_id", None)
            labels = samples.get("label", None)
            if gt_masks is None:
                return {
                    "loss": outputs.loss,
                    "ce_loss": outputs.loss,
                    "mask_bce_loss": 0,
                    "mask_dice_loss": 0,
                    "mask_loss": 0,
                }
            # remember to drop image token
            image_token_length = inputs_llm.shape[1]
            last_hidden_states = outputs.hidden_states[-1][:, image_token_length:, :]

            output_ids = llm_tokens["input_ids"][:, :]
            pred_masks = self.predict_mask(
                output_ids,
                last_hidden_states,
                points,
                shape_id,
            )
        # loss
        ce_loss = outputs.loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            label = labels[batch_idx]
            if self.uselabelratio is True and label in self.label_ratio:
                labelr = self.label_ratio[label]
            else:
                labelr = 1.0

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += labelr * (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += labelr * (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    # Generate
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1,
    ):
        questions = samples["question"]

        if isinstance(questions, str):
            questions = [questions]

        bs = len(questions)
        new_questions = []
        for i in range(bs):
            res = self.prepare_input(question=questions[i])
            new_questions.append(res)

        questions = new_questions

        self.llm_tokenizer.padding_side = "left"
        llm_tokens = self.llm_tokenizer(
            questions, padding="longest", return_tensors="pt"
        ).to(self.device)

        # with self.maybe_autocast(torch.bfloat16):
        with self.maybe_autocast(self.mix_precision):
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            attention_mask = llm_tokens.attention_mask

            if "points" in samples:
                points = samples["points"]
                masks = samples.get("masks", None)
                shape_id = samples.get("shape_id", None)
                inputs_llm, atts_llm, points_pos = self.encode_point(points)

                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            pred_masks = []
            if "points" in samples:
                output_ids = outputs.sequences[:, 1:-1]
                last_hidden_states = [
                    token_states[-1] for token_states in outputs.hidden_states[1:]
                ]
                if len(last_hidden_states) == 0:
                    return {"text": "", "masks": []}
                last_hidden_states = torch.concat(last_hidden_states, dim=1)
                pred_masks = self.predict_mask(
                    output_ids,
                    last_hidden_states,
                    points,
                    shape_id,
                )
        try:
            output_text = self.llm_tokenizer.batch_decode(
                outputs["sequences"], skip_special_tokens=True
            )
        except Exception as e:
            print(outputs)
            raise e
        output_text = [text.strip() for text in output_text]
        masks_score = [score.sigmoid() for score in pred_masks]
        # pro_pred_masks = [(m > 0.4).to(torch.float32) for m in masks_score]
        pro_pred_masks = [(m > 0).to(torch.float32) for m in pred_masks]
        output_dict = {
            "text": output_text,
            "masks_scores": masks_score,
            "masks": pro_pred_masks,
            "output_ids": output_ids,
            "attentions": outputs.attentions,
            "seg_id": self.seg_token_id,
        }

        return output_dict

    # Config and Build Model Function
    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("Load Checkpoint From %s" % url_or_filename)
        return msg

    @classmethod
    def from_config(cls, cfg):
        # Set the mix precision in forward and generate
        mix_precision = cfg.get("mix_precision", "bf16")
        # The seg point Feature dimension
        prompt_encoder_dim = cfg.get("prompt_encoder_dim", 32)
        # Dice_loss_weight
        ce_loss_weight = cfg.get("ce_loss_weight", 1.0)
        bce_loss_weight = cfg.get("bce_loss_weight", 1.0)
        dice_loss_weight = cfg.get("dice_loss_weight", 1.0)
        uselabelratio = cfg.get("uselabelratio", False)
        # Point_encoder set
        point_model_config_path = cfg.get(
            "point_model_config_path",
            "/workspace/project/Research_3D_Aff/3D_ADLLM/configs/models/PointTransformer_2048point.yaml",
        )
        freeze_point = cfg.get("freeze_point", True)
        # seg_encoder
        free_seg_point_encoder = cfg.get("free_seg_point_encoder", False)
        seg_point_encoder_config_path = cfg.get(
            "seg_point_encoder_config_path",
            "/workspace/project/Research_3D_Aff/3D_ADLLM/models/openad/config/PT_modify.py",
        )
        seg_point_encoder_path = cfg.get(
            "seg_point_encoder_path",
            None,
        )
        # aff_decoder
        aff_path = cfg.get("aff_path", None)
        train_aff_decoder = cfg.get("train_aff_decoder", False)
        upscale_points = 2048
        # Lora
        lora_r = cfg.get("lora_r", 16)
        lora_alpha = cfg.get("lora_alpha", 32)
        target_modules = cfg.get("target_modules", ["qkv_proj"])
        # LLM
        llm_model = cfg.get("llm_model", None)
        freeze_llm = cfg.get("freeze_llm", True)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 128)
        freeze_linear = cfg.get("freeze_linear", False)
        label_ratio_path = cfg.get(
            "label_ratio_path",
            "/workspace/project/Research_3D_Aff/3D_ADLLM/result_ratio.json",
        )
        lora_llm_finetune = cfg.get("lora_llm_finetune", False)

        model = cls(
            mix_precision=mix_precision,
            prompt_encoder_dim=prompt_encoder_dim,
            ce_loss_weight=ce_loss_weight,
            dice_loss_weight=dice_loss_weight,
            bce_loss_weight=bce_loss_weight,
            uselabelratio=uselabelratio,
            point_model_config_path=point_model_config_path,
            freeze_point=freeze_point,
            free_seg_point_encoder=free_seg_point_encoder,
            seg_point_encoder_config_path=seg_point_encoder_config_path,
            seg_point_encoder_path=seg_point_encoder_path,
            aff_path=aff_path,
            train_aff_decoder=train_aff_decoder,
            upscale_points=upscale_points,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            llm_model=llm_model,
            freeze_llm=freeze_llm,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            freeze_linear=freeze_linear,
            label_ratio_path=label_ratio_path,
            lora_llm_finetune=lora_llm_finetune,
        )
        model.load_checkpoint_from_config(cfg)
        return model
