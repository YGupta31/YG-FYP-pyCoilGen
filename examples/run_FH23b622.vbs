Set FastHenry2 = CreateObject("FastHenry2.Document")
couldRun = FastHenry2.Run("C:\Users\yashi\OneDrive - Imperial College London\Year 4\MECH70007 - FYP\FYP\YG-FYP-pyCoilGen\examples\coil_track_FH2_input_3b622.inp -S 3b622")
Do While FastHenry2.IsRunning = True
  Wscript.Sleep 500
Loop
inductance = FastHenry2.GetInductance()
FastHenry2.Quit
Set FastHenry2 = Nothing
