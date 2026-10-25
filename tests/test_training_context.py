import numpy as np
import pytest
from scripts.temporal_data import audit_augmentation_row,apply_channel_colour
from kotoba_lipsync.normalization import spectral_residuals


def test_augmentation_cannot_take_a_heldout_voice_or_change_its_owner():
    source={"id":"a","split":"train","language":"ja","speaker_hash":"speaker","transcript_hash":"sentence"}
    augmented={**source,"id":"speed_a","source_recording_id":"a","augmentation":{"kind":"waveform-speed-v1"}}
    audit_augmentation_row(augmented,{"a":source})
    with pytest.raises(ValueError,match="training"):audit_augmentation_row(augmented,{"a":{**source,"split":"validation"}})
    with pytest.raises(ValueError,match="speaker_hash"):audit_augmentation_row({**augmented,"speaker_hash":"other"},{"a":source})


def test_channel_colour_keeps_speaker_normalized_residual_and_level():
    base=np.random.default_rng(52).normal(size=(150,78)).astype(np.float32);base[:,68]=-.4;base[:,75]=.8
    transformed=apply_channel_colour(base,np.linspace(-3,3,40))
    np.testing.assert_allclose(spectral_residuals(transformed,"cmn-v2"),spectral_residuals(base,"cmn-v2"),atol=3e-6)
    np.testing.assert_array_equal(transformed[:,68:],base[:,68:])


def test_event_can_follow_a_training_voice_without_zero_cmn_shortcut(monkeypatch):
    from scripts.train_temporal import SequenceDataset
    class Corpus:
        metadata={"feature_mode":"cmn-v2"}
        mean=np.zeros(158,dtype=np.float32);scale=np.ones(158,dtype=np.float32)
        base=np.zeros((32,78),dtype=np.float32);cmn=np.zeros((32,40),dtype=np.float16)
        base[:16,:40]=2;base[:16,68]=-.4;base[:16,75]=.8
        rows=[{"split":"train","language":"ja","offset":0,"frames":16},{"split":"train","language":"breath","offset":16,"frames":16}]
        def arrays(self,row):
            return self.base[row["offset"]:row["offset"]+16],np.full(16,10 if row["language"]=="ja" else 15),np.full(16,0 if row["language"]=="ja" else 1)
    monkeypatch.setattr(np.random,"random",lambda:0.)
    monkeypatch.setattr(np.random,"randint",lambda *args:0)
    dataset=SequenceDataset(Corpus(),0,0,frames=8,context_augment=True)
    values,y,br,valid,language=dataset[2]
    np.testing.assert_allclose(values.numpy()[:,-40:],-2)
    assert (br.numpy()==1).all() and valid.numpy().all()


def test_random_crop_keeps_future_input_and_target_labels_aligned(monkeypatch):
    from scripts.train_temporal import SequenceDataset
    class Corpus:
        mean=np.zeros(118,dtype=np.float32);scale=np.ones(118,dtype=np.float32)
        rows=[{"split":"train","language":"ja","frames":64}]
        def arrays(self,row):
            base=np.zeros((64,78),dtype=np.float32);base[:,0]=np.arange(64)
            return base,np.arange(64)%16,np.zeros(64,dtype=np.float32)
    monkeypatch.setattr(np.random,"random",lambda:0.)
    monkeypatch.setattr(np.random,"randint",lambda *args:5)
    dataset=SequenceDataset(Corpus(),lookahead=2,context=3,frames=8,random_crop=True)
    x,y,br,valid,language=dataset[0]
    np.testing.assert_array_equal(x.numpy()[:,0],np.arange(2,13))
    np.testing.assert_array_equal(y.numpy()[3:],np.arange(3,11)%16)
    assert (y.numpy()[:3]==-100).all()
