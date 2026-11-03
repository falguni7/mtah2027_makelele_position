"""Who has the ball: rules and models for H / A (roadmap steps 2-4).

  baseline.py      the official nearest-player baseline (vectorised) and carry_forward(), which
                   fills D frames with the last team seen in the half
  control.py       step 2: control events (a player clearly in control of the ball) and the
                   possession they imply, changing at the start of the new team's event
  shape_model.py   step 4: team in possession from the players' positions alone, trained on
                   pseudo-labels from the control rule; used as evidence inside ball gaps

    from mtah.possession.control import control_events, possession_from_events
    events = control_events(match)                         # [(start, stop, team)]
    labels = possession_from_events(events, match.graded)  # "H" / "A" / "D" per frame
"""
