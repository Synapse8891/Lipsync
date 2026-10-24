import json
import time
import numpy as np
import pytest
from kotoba_lipsync.features import FeatureConfig
from scripts.prepare_training import balanced_schedule,PIPELINE_VERSION
from scripts.run_training import plan_commands
from scripts.train_search import iter_batches,choose_trial,run_search

def profile():
    return {"ja_shards":8,"ja_train":8000,"en_train":1200,"ja_eval":300,"en_eval":100,"hours":5,"preparation_fraction":.7,"index_name":"index_five_hour.json"}

def test_five_hour_plan_preserves_training_time_and_separate_index():
    commands,deadline=plan_commands("five-hour",profile(),started=1000)
    assert deadline==19000
    prepare=commands[1]
    assert float(prepare[prepare.index("--deadline")+1])==13600
    assert prepare[prepare.index("--index-name")+1]=="index_five_hour.json"
    assert commands[-1][0]=="scripts/train_search.py"
    assert float(commands[-1][-1])==18880

def test_skip_flags_do_not_create_data_jobs():
    commands,_=plan_commands("five-hour",profile(),skip_download=True,skip_prepare=True,started=1000)
    assert len(commands)==1
    assert commands[0][0]=="scripts/train_search.py"

def test_balanced_prefix_has_both_training_languages_and_no_missing_records():
    records=[("ja","test","jt"),("en","validation","ev")]
    records += [("ja","train",f"j{i}") for i in range(80)]
    records += [("en","train",f"e{i}") for i in range(12)]
    actual=balanced_schedule(records)
    assert actual[:2]==records[:2]
    assert {r[0] for r in actual[2:48]}=={"ja","en"}
    assert sorted(actual)==sorted(records)

def make_cache(tmp_path):
    records=[]
    bases={}
    for lang in ("ja","en"):
        name=lang+"_train"
        base=np.arange(16*78,dtype=np.float32).reshape(16,78)/1000
        labels=np.array([0,0,10,10]*4,dtype=np.int16)
        np.savez(tmp_path/f"{name}.npz",base=base,y=labels,annotation_signature=lang,pipeline_version=PIPELINE_VERSION)
        records.append({"id":name,"language":lang,"split":"train","speaker_hash":name,"transcript_hash":name})
        bases[lang]=base
    index={"records":records,"annotation_signatures":{"ja":"ja","en":"en"},"pipeline_version":PIPELINE_VERSION,"teacher_preprocessing":{"ja":{"do_normalize":False},"en":{"do_normalize":True}},"label_kind":"synthetic test fixture"}
    return index,bases

def test_stream_batches_have_correct_lookahead_alignment_and_keep_all_frames(tmp_path):
    index,bases=make_cache(tmp_path)
    cfg=FeatureConfig(lookahead_frames=3)
    batches=list(iter_batches(index,"train",cfg,feature_dir=tmp_path,batch_frames=5,stride=2))
    assert sum(len(y) for _,y,_ in batches)==14
    x,y,ll=batches[0]
    t=np.arange(0,13,2)
    expected=np.concatenate([bases["ja"][np.maximum(t+3-delay,0)] for delay in cfg.context],axis=1)
    np.testing.assert_array_equal(x,expected)
    assert set(ll)=={"ja"}
    assert len(x)<5+16

def test_expired_budget_does_not_open_feature_files(tmp_path,monkeypatch):
    index,_=make_cache(tmp_path)
    monkeypatch.setattr(np,"load",lambda *a,**k:pytest.fail("expired job opened feature data"))
    with pytest.raises(TimeoutError):
        list(iter_batches(index,"train",FeatureConfig(),feature_dir=tmp_path,deadline=time.time()-1))

def test_candidate_selection_ignores_test_scores_and_breaks_ties_by_size():
    results=[{"trial":1,"selection_score":.7,"parameter_count":100,"test_accuracy":1.0},
             {"trial":2,"selection_score":.8,"parameter_count":200,"test_accuracy":.1},
             {"trial":3,"selection_score":.8,"parameter_count":150,"test_accuracy":.05}]
    assert choose_trial(results)["trial"]==3

def test_search_does_not_read_test_data(tmp_path,monkeypatch):
    from scripts import train_search as search
    index,_=make_cache(tmp_path)
    for lang in ("ja","en"):
        source=tmp_path/f"{lang}_train.npz"
        (tmp_path/f"{lang}_validation.npz").write_bytes(source.read_bytes())
        index["records"].append({"id":lang+"_validation","language":lang,"split":"validation","speaker_hash":lang+"_v","transcript_hash":lang+"_v"})
        # Deliberately absent files must not be opened during model selection.
        index["records"].append({"id":lang+"_forbidden_test","language":lang,"split":"test","speaker_hash":lang+"_t","transcript_hash":lang+"_t"})
    class FakeClassifier:
        def __init__(self,**kwargs):pass
        def partial_fit(self,x,y,classes,sample_weight):
            self.coefs_=[np.zeros((x.shape[1],2),dtype=np.float32)]
            self.intercepts_=[np.array([1,0],dtype=np.float32)]
        def predict(self,x):return np.zeros(len(x),dtype=np.int16)
    monkeypatch.setattr(search,"MLPClassifier",FakeClassifier)
    # No actual optimization is run; the fake classifier only supplies fixed weights.
    winner,trials=run_search(index,{"trials":[{"hidden":[2],"seed":42}]},FeatureConfig(),time.time()+30,epochs=1,feature_dir=tmp_path,work_dir=tmp_path/"work")
    assert winner["trial"]==1
    assert len(trials)==1

def test_dry_run_never_starts_jobs_or_writes_status(tmp_path,monkeypatch,capsys):
    from scripts import run_training as runner
    config_dir=tmp_path/"configs";config_dir.mkdir()
    (config_dir/"training_profiles.json").write_text(json.dumps({"five-hour":profile()}))
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner.sys,"argv",["run_training.py","--profile","five-hour","--dry-run"])
    monkeypatch.setattr(runner.subprocess,"run",lambda *a,**k:pytest.fail("dry run launched a subprocess"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["budget_hours"]==5
    assert not (tmp_path/"reports").exists()
