from collections import Counter
from scripts.extend_cv22_training import select_candidates
from scripts.prepare_training import key


def row(ident,client,text):
    return {"path":ident+".mp3","client_id":client,"sentence":text,"up_votes":"3","down_votes":"0","source_split":"test"}


def test_cross_version_selection_excludes_global_speakers_ids_and_text():
    previous=[{"id":"old","split":"train","speaker_hash":key("jaexisting"),"transcript_hash":key("jaused text")}]
    reserved=[{"id":"heldout","split":"test","speaker_hash":key("enprotected"),"global_speaker_hash":key("protected"),"transcript_hash":key("enheldout")}]
    rows=[row("old","other","new 1"),row("new_1","protected","new 2"),row("new_2","other","used text"),row("new_3","existing","valid text"),row("new_4","fresh","another text"),row("new_5","","empty client")]
    rows.append({**row("new_6","voter","rejected"),"down_votes":"4"})
    result=select_candidates(rows,previous,reserved,set(),20)
    assert set(result)=={"new_3","new_4"}
    assert all(r["global_speaker_hash"]!=key("protected") for r in result.values())


def test_selection_balances_speakers_and_avoids_duplicate_sentences():
    rows=[row("a"+str(i),"a","text a "+str(i)) for i in range(5)]+[row("b"+str(i),"b","text b "+str(i)) for i in range(5)]
    rows.append(row("duplicate","c","text a 0"))
    result=select_candidates(rows,[],[],set(),4)
    counts=Counter(r["speaker_hash"] for r in result.values())
    assert len(counts)>=2 and max(counts.values())<=2
    assert len({r["transcript_hash"] for r in result.values()})==4


def test_fallback_global_protection_from_older_metadata_is_enforced():
    result=select_candidates([row("new","protected_old","unseen")],[],[],{key("protected_old")},10)
    assert not result
