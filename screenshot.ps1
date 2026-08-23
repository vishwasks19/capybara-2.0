Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$Screen = [System.Windows.Forms.SystemInformation]::VirtualScreen
$Width  = $Screen.Width
$Height = $Screen.Height
$Left   = $Screen.Left
$Top    = $Screen.Top

$Bitmap = New-Object System.Drawing.Bitmap $Width, $Height
$Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
$Graphics.CopyFromScreen($Left, $Top, 0, 0, $Bitmap.Size)
$Bitmap.Save("c:\Users\PARTHA\Desktop\coding\Github\Capybara2.0\figma_screenshot.png")
$Graphics.Dispose()
$Bitmap.Dispose()
