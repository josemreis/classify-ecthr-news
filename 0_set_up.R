### wd at file location
parent_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(parent_dir)

### dirs
if (!dir.exists("scripts")) {
  
  dir.create("scripts")
  
  if (!dir.exists("scripts/helpers")) {
    
    dir.create("scripts/helpers")
    
  }
  
}


if (!dir.exists("data")) {
  
  dir.create("data")
  
}

### Move deccano api helpers script to helpers folder
## re-run at change
file.copy(
  from = '/home/jmr/Dropbox/python_default/doccano_api_helpers/scripts/doccano_helpers.py',
  to = 'scripts/helpers/doccano_helpers.py'
)
