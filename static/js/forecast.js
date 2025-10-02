const API = "http://localhost:8000/api/forecast";
fetch(API).then(r=>r.json()).then(d=>{
  city.textContent=d.city; aqi.textContent=d.aqi; ts.textContent=d.ts;
  const badge=status; badge.textContent=d.status;
  let cls="bg-success";
  if(d.aqi>200) cls="bg-danger"; else if(d.aqi>150) cls="bg-warning";
  else if(d.aqi>100) cls="bg-primary"; else if(d.aqi>50) cls="bg-info";
  badge.className="badge "+cls;
}).catch(console.error);
