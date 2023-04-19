from transformers import TextClassificationPipeline

class CustomTextClassificationPipeline(TextClassificationPipeline):
    def _forward(self, model_inputs, **forward_params):
        if "input_ids" in model_inputs and isinstance(model_inputs["input_ids"], dict):
            return self.model(**model_inputs["input_ids"])
        return super()._forward(model_inputs, **forward_params)
