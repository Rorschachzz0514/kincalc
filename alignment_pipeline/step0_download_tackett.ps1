# Step 0-B  下载 Tackett《唐五代人物传记与社会网络资料库》并把 Access 表导出为 CSV
#
# 来源: https://history.berkeley.edu/nicolas-tackett  (open access, Google Drive 链接)
#   v1.0  tbdb010.mdb    Google Drive id 1fZnfXLkUv6njoQOmhm_o6cEtlmPGj46H
#   v1.5  tbdb015.accdb  Google Drive id 1wbRWFmR3IwuwMXykTjVFjzvrLQqD_eBK
# Google Drive 对 >几 MB 的文件要先拿 confirm/uuid 再下载, 下面用两次请求完成。
# 导出用 Windows 自带的 ACE OLEDB 驱动 (装了 Office 即有), 每张表一个 CSV (UTF-8)。
#
# 用法:  powershell -ExecutionPolicy Bypass -File step0_download_tackett.ps1

$dir = Join-Path $PSScriptRoot "..\data\external\1_史学家人工关系库\Tackett_TBDB"
New-Item -ItemType Directory -Force $dir | Out-Null
$files = @{ "tbdb010_v1.0.mdb" = "1fZnfXLkUv6njoQOmhm_o6cEtlmPGj46H"; "tbdb015_v1.5.accdb" = "1wbRWFmR3IwuwMXykTjVFjzvrLQqD_eBK" }

foreach ($name in $files.Keys) {
  $out = Join-Path $dir $name
  if (Test-Path $out) { "已存在, 跳过 $name"; continue }
  $id = $files[$name]
  $sess = New-Object Microsoft.PowerShell.Commands.WebRequestSession
  $r = Invoke-WebRequest -UseBasicParsing -WebSession $sess "https://drive.google.com/uc?export=download&id=$id"
  $uuid = [regex]::Match($r.Content, 'name="uuid" value="([^"]+)"').Groups[1].Value
  Invoke-WebRequest -UseBasicParsing -WebSession $sess -OutFile $out "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t&uuid=$uuid"
  "下载完成 $name  $((Get-Item $out).Length) bytes"
}

foreach ($name in $files.Keys) {
  $p = Join-Path $dir $name
  $csvdir = Join-Path $dir (($name -split "_")[0] + "_csv")
  New-Item -ItemType Directory -Force $csvdir | Out-Null
  $cn = New-Object System.Data.OleDb.OleDbConnection("Provider=Microsoft.ACE.OLEDB.16.0;Data Source=$p;")
  try { $cn.Open() } catch { $cn = New-Object System.Data.OleDb.OleDbConnection("Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$p;"); $cn.Open() }
  foreach ($t in ($cn.GetSchema("Tables") | Where-Object { $_.TABLE_TYPE -eq "TABLE" })) {
    $cmd = $cn.CreateCommand(); $cmd.CommandText = "SELECT * FROM [$($t.TABLE_NAME)]"
    $dt = New-Object System.Data.DataTable; [void](New-Object System.Data.OleDb.OleDbDataAdapter $cmd).Fill($dt)
    $dt | Export-Csv -Path (Join-Path $csvdir "$($t.TABLE_NAME).csv") -NoTypeInformation -Encoding UTF8
    "{0}: {1,-28} {2,6} 行" -f $name, $t.TABLE_NAME, $dt.Rows.Count
  }
  $cn.Close()
}
