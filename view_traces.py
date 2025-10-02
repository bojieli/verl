#!/usr/bin/env python3
"""
Helper script to view MLflow traces from saved JSON files.
Usage: python view_traces.py [--trace-dir mlruns/1/traces] [--filter step=1] [--limit 10]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def load_trace(trace_path: Path) -> Dict[str, Any]:
    """Load a trace from its JSON file."""
    trace_file = trace_path / "artifacts" / "traces.json"
    if trace_file.exists():
        with open(trace_file, 'r') as f:
            return json.load(f)
    return {}


def extract_trace_info(trace_data: Dict) -> Dict[str, Any]:
    """Extract key information from trace data."""
    if not trace_data:
        return {}
    
    info = {
        "trace_id": trace_data.get("trace_id"),
        "status": trace_data.get("status", {}).get("code"),
        "spans": len(trace_data.get("spans", [])),
    }
    
    # Extract tags from first span if available
    spans = trace_data.get("spans", [])
    if spans:
        first_span = spans[0]
        attributes = first_span.get("attributes", {})
        
        # Look for common tags
        for key in ["mlflow.spanInputs", "mlflow.spanOutputs"]:
            if key in attributes:
                try:
                    data = json.loads(attributes[key])
                    if key == "mlflow.spanInputs":
                        # Extract index, step, etc from kwargs
                        kwargs = data.get("kwargs", {})
                        info["index"] = kwargs.get("index")
                        info["data_source"] = kwargs.get("data_source")
                        extra = kwargs.get("extra_info", {})
                        if isinstance(extra, dict):
                            info["sample_index"] = extra.get("index")
                    elif key == "mlflow.spanOutputs":
                        # Get token count or response length
                        token_ids = data.get("token_ids", [])
                        info["output_tokens"] = len(token_ids) if isinstance(token_ids, list) else None
                except (json.JSONDecodeError, AttributeError):
                    pass
    
    return info


def list_traces(trace_dir: Path, filters: Dict[str, str] = None, limit: int = None) -> List[Dict]:
    """List all traces with optional filtering."""
    traces = []
    
    trace_folders = sorted([d for d in trace_dir.iterdir() if d.is_dir() and d.name.startswith("tr-")])
    
    for trace_path in trace_folders:
        if limit and len(traces) >= limit:
            break
            
        trace_data = load_trace(trace_path)
        info = extract_trace_info(trace_data)
        
        if not info:
            continue
        
        # Apply filters
        if filters:
            match = all(
                str(info.get(k)) == v for k, v in filters.items()
            )
            if not match:
                continue
        
        info["path"] = str(trace_path)
        traces.append(info)
    
    return traces


def view_trace_detail(trace_path: Path):
    """View detailed information about a specific trace."""
    trace_data = load_trace(trace_path)
    
    if not trace_data:
        print("No trace data found")
        return
    
    print(f"Trace ID: {trace_data.get('trace_id')}")
    print(f"Status: {trace_data.get('status', {}).get('code')}")
    print(f"\nSpans ({len(trace_data.get('spans', []))}):")
    
    for i, span in enumerate(trace_data.get("spans", []), 1):
        print(f"\n  [{i}] {span.get('name')}")
        attributes = span.get("attributes", {})
        
        # Show inputs
        if "mlflow.spanInputs" in attributes:
            try:
                inputs = json.loads(attributes["mlflow.spanInputs"])
                print(f"      Inputs: {json.dumps(inputs, indent=8)[:500]}...")
            except:
                pass
        
        # Show outputs
        if "mlflow.spanOutputs" in attributes:
            try:
                outputs = json.loads(attributes["mlflow.spanOutputs"])
                print(f"      Outputs: {json.dumps(outputs, indent=8)[:500]}...")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="View MLflow traces from saved JSON files")
    parser.add_argument("--trace-dir", default="mlruns/1/traces", help="Directory containing traces")
    parser.add_argument("--filter", action="append", help="Filter traces (key=value)")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of traces to show")
    parser.add_argument("--detail", help="Show detailed view of specific trace (provide trace folder name)")
    parser.add_argument("--count", action="store_true", help="Just count traces")
    
    args = parser.parse_args()
    
    trace_dir = Path(args.trace_dir)
    
    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        return
    
    # Count mode
    if args.count:
        trace_folders = [d for d in trace_dir.iterdir() if d.is_dir() and d.name.startswith("tr-")]
        print(f"Total traces: {len(trace_folders)}")
        return
    
    # Detail mode
    if args.detail:
        trace_path = trace_dir / args.detail
        if not trace_path.exists():
            print(f"Error: Trace not found: {trace_path}")
            return
        view_trace_detail(trace_path)
        return
    
    # List mode
    filters = {}
    if args.filter:
        for f in args.filter:
            if '=' in f:
                k, v = f.split('=', 1)
                filters[k] = v
    
    traces = list_traces(trace_dir, filters=filters, limit=args.limit)
    
    print(f"Found {len(traces)} traces:\n")
    print(f"{'Trace ID':<40} {'Spans':<8} {'Output Tokens':<15} {'Data Source':<15}")
    print("-" * 100)
    
    for trace in traces:
        trace_id = trace.get("trace_id", "N/A")[:38]
        spans = trace.get("spans", "N/A")
        tokens = trace.get("output_tokens") or "N/A"
        source = trace.get("data_source") or "N/A"
        
        print(f"{trace_id:<40} {spans:<8} {tokens!s:<15} {source:<15}")
    
    if traces:
        print(f"\nTo view details: python view_traces.py --detail {Path(traces[0]['path']).name}")


if __name__ == "__main__":
    main()

