import json
import numpy as np
import pytest


def test_numeric_trees_match_native_multiclass_residuals_with_missing_values(tmp_path):
    xgb=pytest.importorskip("xgboost")
    from kotoba_lipsync.tree_decoder import pack_numeric_trees,TreeResidual
    rng=np.random.default_rng(63);x=rng.normal(size=(300,7)).astype(np.float32);x[::11,2]=np.nan
    y=np.arange(len(x))%16;baseline=rng.normal(0,.2,(len(x),16)).astype(np.float32)
    matrix=xgb.DMatrix(x,label=y,base_margin=baseline)
    model=xgb.train({"objective":"multi:softprob","num_class":16,"max_depth":3,"nthread":2,"seed":63},matrix,num_boost_round=8)
    path=tmp_path/"trees.json";model.save_model(path)
    arrays,config=pack_numeric_trees(json.loads(path.read_text()))
    predictor=TreeResidual(arrays,config);expected=model.predict(matrix,output_margin=True)-baseline
    np.testing.assert_allclose(predictor.predict(x),expected,atol=2e-6)
    np.testing.assert_allclose(predictor.predict(x[0]),expected[0],atol=2e-6)
    assert config["max_depth"]<=3
