import json
import os

import ezkl
import torch
import torch.onnx
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.AvgPool2d(2, 1, (1, 1))

    def forward(self, x):
        return self.layer(x)[0]


async def main():
    circuit = MyModel()
    x = 0.1 * torch.rand(1, *[3, 2, 2], requires_grad=True)

    # Flips the neural net into inference mode
    circuit.eval()

    # Export the model
    torch.onnx.export(
        circuit,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "network.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    data_array = ((x).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_data=[data_array])

    # Serialize data into file:
    json.dump(data, open("input.json", "w"))

    model_path = os.path.join("model.onnx")
    compiled_model_path = os.path.join("model.compiled")
    pk_path = os.path.join("pk.key")
    vk_path = os.path.join("vk.key")
    settings_path = os.path.join("settings.json")
    srs_path = os.path.join("kzg.srs")
    data_path = os.path.join("input.json")

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "public"
    run_args.param_visibility = "private"
    run_args.output_visibility = "public"
    run_args.num_inner_cols = 1
    run_args.variables = [("batch_size", 1)]

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    assert res == True

    # generate a bunch of dummy calibration data
    cal_data = {
        "input_data": [(0.1 * torch.rand(2, *[3, 2, 2])).flatten().tolist()],
    }

    cal_path = os.path.join("val_data.json")
    # save as json file
    with open(cal_path, "w") as f:
        json.dump(cal_data, f)

    res = await ezkl.calibrate_settings(
        cal_path, model_path, settings_path, "resources"
    )

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = await ezkl.get_srs(settings_path)

    # setup keypair
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    witness_path = "witness.json"

    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path, vk_path)
    assert os.path.isfile(witness_path)

    # we force the output to be 1 this corresponds to the solvency test being true -- and we set this to a fixed vis output
    # this means that the output is fixed and the verifier can see it but that if the input is not in the set the output will not be 0 and the verifier will reject
    witness = json.load(open(witness_path, "r"))
    witness["outputs"][0] = [ezkl.float_to_felt(1.0, 0)]
    json.dump(witness, open(witness_path, "w"))

    proof_path = os.path.join("proof.json")
    # proof path
    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
    )

    assert os.path.isfile(proof_path)

    print(res)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
