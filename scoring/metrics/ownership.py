def check_institutional_ownership(info):
    io = info.get("heldPercentInstitutions", None)
    if io is None:
        return 0, None
    if io > 0.5:
        return 1, f"✅ High institutional ownership {io*100:.1f}%"
    return 0, f"❌ Low institutional ownership {io*100:.1f}%"

def check_insider_ownership(info):
    ins = info.get("heldPercentInsiders", None)
    if ins is None:
        return 0, None
    if ins > 0.01:
        return 1, f"✅ Insiders own {ins*100:.1f}%"
    return 0, f"❌ Low insider ownership {ins*100:.1f}%"