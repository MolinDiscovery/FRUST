# frust/utils/mols.py

def get_molecule_name(smiles: str):
    """Retrieve the IUPAC name for a molecule from its SMILES string.
    
    Queries the PubChem database to get the IUPAC name for a given molecule.
    Implements retry logic to handle server busy errors. Returns a sanitized
    name with spaces replaced by underscores for file naming compatibility.
    
    Args:
        smiles (str): The SMILES representation of the molecule.
        
    Returns:
        str: The IUPAC name with spaces replaced by underscores, or 
             "Unknown_Molecule" if retrieval fails or no name is found.
             
    Raises:
        pcp.PubChemHTTPError: When PubChem API encounters errors other than 
                             server busy status (which is handled with retries).
    """
    import pubchempy as pcp
    import time

    max_retries = 5      # Maximum number of retry attempts
    delay = 10           # Seconds to wait between attempts

    compounds = None
    for attempt in range(1, max_retries + 1):
        try:
            compounds = pcp.get_compounds(smiles, 'smiles')
            break
        except pcp.PubChemHTTPError as e:
            if 'PUGREST.ServerBusy' in str(e):
                print(f"Attempt {attempt}/{max_retries}: PubChem is busy. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print(f"Attempt {attempt}/{max_retries}: Encountered error: {e}")
                break

    if compounds and compounds[0].iupac_name:
        name = compounds[0].iupac_name
        return name.replace(" ", "_")
    else:
        print("Warning: Failed to retrieve compound name. Using fallback name.")
        return "Unknown_Molecule"
    

def generate_id(id_name: str, job_id:int = None) -> str:
    """Generate a unique identifier string with timestamp and hash.
    
    Creates a unique ID by combining a base name, optional job ID, current 
    timestamp, and a short UUID hash. The format ensures chronological 
    ordering while maintaining uniqueness.
    
    Args:
        id_name (str): Base name for the identifier.
        job_id (int, optional): Optional job identifier to include in the ID.
                               If provided, will be formatted as "job{job_id}".
                               Defaults to None.
        
    Returns:
        str: Formatted unique identifier string. Format is either:
             "{id_name}-job{job_id}-{YYMMDD-HHMMSS}-{hash}" if job_id provided,
             or "{id_name}-{YYMMDD-HHMMSS}-{hash}" if job_id is None.
             
    Example:
        >>> generate_id("analysis", 123)
        'analysis-job123-250527-142530-a4'
        >>> generate_id("calculation")
        'calculation-250527-142530-b7'
    """
    import datetime
    import uuid
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%y%m%d-%H%M%S')  # YYMMDD-HHMMSS
    # Create a UUID and take the first 2 characters for a shorter hash
    uuid_hash = str(uuid.uuid4())[:2]
    if job_id:
        id = f"{id_name}-job{job_id}-{formatted_time}-{uuid_hash}"
    else:
        id = f"{id_name}-{formatted_time}-{uuid_hash}"
    return id


def get_rpos_from_name(ts_name):
    import re
    match = re.search(r"rpos\((\d+)\)", ts_name)
    if match:
        return int(match.group(1))
    else:
        print("No rpos found in name!")


