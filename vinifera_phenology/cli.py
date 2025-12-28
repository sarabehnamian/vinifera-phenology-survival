"""
Command-line interface for vinifera_phenology package.
"""

import sys
from pathlib import Path
from . import survival_analysis


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: vinifera-survival <input_csv> [output_xlsx]")
        print("\nExample:")
        print("  vinifera-survival data/observations.csv output/intervals.xlsx")
        sys.exit(1)
    
    input_csv = Path(sys.argv[1])
    output_xlsx = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        results = survival_analysis.process_phenology_data(
            csv_path=input_csv,
            output_path=output_xlsx
        )
        
        print(f"\nProcessed {len(results)} phenophases:")
        for pheno, df in results.items():
            if not df.empty:
                print(f"  {pheno}: {len(df)} intervals")
        
        if output_xlsx:
            print(f"\nResults saved to: {output_xlsx}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

