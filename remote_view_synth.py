import subprocess

def run(**args):
    print("Starte Subprocess")
    process = subprocess.Popen([args["cmd"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()

    print_out(stderr)
    print_out(stdout)


def print_out(out, only_last=False):
    # Ausgabe decodieren und in Zeilen aufteilen
    output_lines = out.decode().splitlines()

    # Letzte Zeile der Ausgabe (letztes print Statement)
    if output_lines :
        if only_last:
            last_print_statement = output_lines[-1]
            print(last_print_statement)
        else:
            for line in output_lines:
                print(line)
    
    

