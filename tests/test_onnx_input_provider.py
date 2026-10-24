"""The ONNX input provider turns an uploaded model file into the session a plugin runs."""
import pytest

from aisc_plugin_interface import OnnxInputProvider


def test_the_provider_makes_a_session_of_the_uploaded_bytes():
    pytest.importorskip("onnx", reason="the test builds its model with onnx")
    pytest.importorskip("onnxruntime", reason="the provider makes its session with onnxruntime")
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "echo",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    session = OnnxInputProvider(model.SerializeToString()).get_data()
    assert [o.name for o in session.get_outputs()] == ["y"]
